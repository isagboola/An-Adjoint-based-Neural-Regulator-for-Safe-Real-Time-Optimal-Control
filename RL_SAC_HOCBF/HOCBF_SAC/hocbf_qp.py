import numpy as np
import torch

from HOCBF_SAC.dynamics import DYNAMICS_MODE
from HOCBF_SAC.utils import prRed, to_tensor


class HOCBFQPLayer:
    def __init__(self, env, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception("Dynamics mode not supported.")
        if self.env.dynamics_mode != "Unicycle":
            raise Exception("Only Unicycle CBF-QP is supported in this project.")

        self.k1 = float(getattr(args, "k1", 3.0))
        self.k2 = float(getattr(args, "k2", 3.0))
        self.robot_r = float(getattr(args, "robot_r", 0.25))
        self.buffer = float(getattr(args, "cbf_buffer", getattr(args, "safety_margin", 0.05)))
        self.solver_backend = str(getattr(args, "qp_solver", "qpth")).lower()
        if self.solver_backend not in ("qpth", "qpoases"):
            raise ValueError("qp_solver must be 'qpth' or 'qpoases'.")
        self._qpoases_build_count = 0

    def get_safe_action(self, state_batch, action_batch):
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)

        Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, action_batch)
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1), self.u_max.repeat(action_batch.shape[0], 1),)
        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        if self.solver_backend == "qpoases":
            sol = self._solve_qpoases(Ps, qs, Gs, hs)
        else:
            sol = self.cbf_layer(
                Ps, qs, Gs, hs,
                solver_args={"check_Q_spd": False, "maxIter": 100000,
                             "notImprovedLim": 10, "eps": 1e-4},
            )
        return sol[:, : self.env.action_space.shape[0]]

    def cbf_layer(self, Qs, ps, Gs, hs, As=None, bs=None, solver_args=None):
        from qpth.qp import QPFunction

        if solver_args is None:
            solver_args = {}
        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)( Qs.double(), ps.double(), Gs.double(), hs.double(), As, bs).float()
        if torch.any(torch.isnan(result)):
            prRed("QP failed to solve (nan result).")
            raise Exception("QP failed to solve.")
        return result

    def _qpoases_solver(self, num_actions, num_constraints):
        """Build a fresh qpOASES solver for the current control step."""
        try:
            import casadi as ca
        except ImportError as exc:
            raise ImportError(
                "The qpOASES backend requires CasADi with the qpoases plugin."
            ) from exc
        structure = {
            "h": ca.Sparsity.dense(num_actions, num_actions),
            "a": ca.Sparsity.dense(num_constraints, num_actions),
        }
        self._qpoases_build_count += 1
        name = (f"hocbf_qpoases_{id(self)}_{num_actions}_{num_constraints}_"
                f"{self._qpoases_build_count}")
        options = {"printLevel": "none", "error_on_fail": False}
        return ca.conic(name, "qpoases", structure, options)

    def _solve_qpoases(self, Qs, ps, Gs, hs):
        """Solve each deployment QP on CPU and return results on the input device."""
        batch_size, num_actions = ps.shape
        num_constraints = Gs.shape[1]
        solver = self._qpoases_solver(num_actions, num_constraints)
        q_np = Qs.detach().cpu().double().numpy()
        p_np = ps.detach().cpu().double().numpy()
        g_np = Gs.detach().cpu().double().numpy()
        h_np = hs.detach().cpu().double().numpy()
        solutions = []
        lower_constraints = np.full(num_constraints, -np.inf)
        for index in range(batch_size):
            result = solver(
                h=q_np[index], g=p_np[index], a=g_np[index],
                lba=lower_constraints, uba=h_np[index],
            )
            solution = np.asarray(result["x"], dtype=float).reshape(num_actions)
            stats = solver.stats()
            if not np.isfinite(solution).all() or not stats.get("success", True):
                status = stats.get("return_status", "unknown")
                raise RuntimeError(f"qpOASES failed to solve the HOCBF QP: {status}")
            solutions.append(solution)
        return torch.as_tensor(
            np.asarray(solutions), dtype=Qs.dtype, device=Qs.device
        )

    def get_cbf_qp_constraints(self, state_batch, action_batch):
        assert (len(state_batch.shape) == 2
            and len(action_batch.shape) == 2
        ), print(state_batch.shape, action_batch.shape)

        batch_size = state_batch.shape[0]

        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)

        num_cbfs = len(self.env.hazards)
        thetas = state_batch[:, 2, :].squeeze(-1)
        vs = state_batch[:, 3, :].squeeze(-1)
        c_thetas = torch.cos(thetas)
        s_thetas = torch.sin(thetas)

        # Position barriers are built on robot position p=[x,y].
        ps = torch.zeros((batch_size, 2)).to(self.device)
        ps[:, 0] = state_batch[:, 0, :].squeeze(-1)
        ps[:, 1] = state_batch[:, 1, :].squeeze(-1)

        hs = 1e3 * torch.ones((batch_size, num_cbfs), device=self.device)
        dhdps = torch.zeros((batch_size, num_cbfs, 2), device=self.device)
        hazards = self.env.hazards
        for i in range(len(hazards)):
            if hazards[i]["type"] != "circle":
                raise Exception(
                    "Only circle obstacles are supported in this project, got: {}".format(
                        hazards[i]["type"]
                    )
                )

            obs_loc = to_tensor(hazards[i]["location"], torch.FloatTensor, self.device)
            px = ps[:, 0] - obs_loc[0]
            py = ps[:, 1] - obs_loc[1]

            # r_eff = r_obs + robot_r + buffer
            # h = (x-xc)^2 + (y-yc)^2 - r_eff^2
            r_eff = hazards[i]["radius"] + self.robot_r + self.buffer
            hs[:, i] = px * px + py * py - r_eff * r_eff
            dhdps[:, i, :] = ps - obs_loc

        n_u = action_batch.shape[1]
        num_constraints = num_cbfs + 2 * n_u
        G = torch.zeros((batch_size, num_constraints, n_u)).to(self.device)
        h = torch.zeros((batch_size, num_constraints)).to(self.device)
        ineq_constraint_counter = 0

        e = torch.zeros((batch_size, 2), device=self.device)
        e[:, 0] = c_thetas
        e[:, 1] = s_thetas
        e_perp = torch.zeros((batch_size, 2), device=self.device)
        e_perp[:, 0] = -s_thetas
        e_perp[:, 1] = c_thetas

        s = torch.sum(dhdps * e.unsqueeze(1), dim=2)  # (batch, num_cbfs)
        s_perp = torch.sum(dhdps * e_perp.unsqueeze(1), dim=2)  # (batch, num_cbfs)
        hdot = 2.0 * vs.unsqueeze(1) * s
        c0 = 2.0 * (vs.unsqueeze(1) ** 2)
        c_a = 2.0 * s
        c_w = 2.0 * vs.unsqueeze(1) * s_perp
        k1 = self.k1
        k2 = self.k2

        a_nom = action_batch[:, 0, 0].unsqueeze(1)
        w_nom = action_batch[:, 1, 0].unsqueeze(1)
        lhs_nom = c_a * a_nom + c_w * w_nom

        rhs = -(c0 + k1 * hdot + k2 * hs)

        # Delta-action form without slack:
        # c_a*(a_nom+da) + c_w*(w_nom+dw) >= rhs
        # <=> -c_a*da - c_w*dw <= -rhs + (c_a*a_nom + c_w*w_nom)
        G[:, :num_cbfs, 0] = -c_a
        G[:, :num_cbfs, 1] = -c_w
        h[:, :num_cbfs] = -rhs + lhs_nom
        ineq_constraint_counter += num_cbfs

        P = torch.diag(torch.tensor([1.0, 1.0])).repeat(batch_size, 1, 1).to(self.device)
        q = torch.zeros((batch_size, n_u)).to(self.device)

        for c in range(n_u):
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self):
        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)
        return u_min, u_max
