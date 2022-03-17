"""
Standard MPC for following a pre-defined trajectory
"""
import casadi as ca
import numpy as np


class MPCSolver(object):
    """
    Nonlinear MPC
    """

    def __init__(self, pred_time_horizon, pred_time_step, so_path="./nmpc.so"):
        """
        Nonlinear MPC for quadrotor control
        """
        self.so_path = so_path

        # Time constant
        self._pred_time_horizon = pred_time_horizon
        self._pred_time_step = pred_time_step
        self._num_pred_steps = int(self._pred_time_horizon / self._pred_time_step)

        # Gravity
        self._gravity = 9.81
        self._mass = 1.0  # 3.2
        # self._mass = 3.2

        # Quadrotor constants
        # self._w_max_yaw = 6.0
        # self._w_max_xy = 6.0
        # self._thrust_min = 2.0
        # self._thrust_max = 20.0

        # parameters from alphapilot:
        self._w_max_yaw = 6.0  # 3.0
        # self._w_max_yaw = 3.0
        # self._w_max_yaw = 8.31
        self._w_max_xy = 6.0  # 3.0
        # self._w_max_xy = 3.0
        # self._w_max_xy = 8.31
        self._thrust_min = 0.5
        self._thrust_max = 21.0  # 16.0
        # self._thrust_max = 16.0 * 4

        # state dimension (px, py, pz,           # quadrotor position
        #                  qw, qx, qy, qz,       # quadrotor quaternion
        #                  vx, vy, vz,           # quadrotor linear velocity
        self._state_dim = 10
        # action dimensions (c_thrust, wx, wy, wz)
        self._action_dim = 4

        # cost matrix for tracking the goal point
        self._cost_matrix_goal = np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, 10,  # delta_qw, delta_qx, delta_qy, delta_qz
            10, 10, 10
        ])

        # cost matrix for following the trajectory
        self._cost_matrix_traj = np.diag([
            200, 200, 500,          # position
            100, 100, 100, 100,     # rotation (as quaternion)
            50, 50, 50,             # velocity
        ])

        """
        self._cost_matrix_traj = np.diag([
            200, 200, 500,              # position
            1000, 1000, 1000, 2000,     # rotation (as quaternion)
            50, 50, 50,                 # velocity
        ])
        """

        """
        self._cost_matrix_traj = np.diag([
            0, 0, 0,  # position
            0, 0, 0, 0,  # rotation (as quaternion)
            0, 0, 0,  # velocity
        ])
        """

        """
        # "pre-multi-trajectory" cost matrix
        self._cost_matrix_traj = np.diag([
            100, 100, 300,  # position
            100, 100, 100, 100,  # rotation (as quaternion)
            50, 50, 50,  # velocity
        ])
        """
        self._cost_matrix_traj = np.diag([
            200, 200, 500,  # position
            50, 50, 50, 50,  # rotation (as quaternion)
            10, 10, 10,  # velocity
        ])

        self._cost_quaternion_norm = 10000

        # cost matrix for the action
        # self._cost_matrix_action = np.diag([0.1, 0.1, 0.1, 0.1])  # T, wx, wy, wz
        self._cost_matrix_action = np.diag([1.0, 1.0, .01, 1.0])  # T, wx, wy, wz
        # self._cost_matrix_action = np.diag([0, 0, 0, 0])
        # self._cost_matrix_action = np.diag([100, 100, 100, 100])  # T, wx, wy, wz

        # initial state and control action
        self._quad_state_0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_action_0 = [9.81, 0.0, 0.0, 0.0]

        self._init_dynamics()

    def _init_dynamics(self):
        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        # position
        pos_x, pos_y, pos_z = ca.SX.sym("pos_x"), ca.SX.sym("pos_y"), ca.SX.sym("pos_z")
        # rotation
        quat_w, quat_x, quat_y, quat_z = ca.SX.sym("quat_w"), ca.SX.sym("quat_x"), ca.SX.sym("quat_y"), ca.SX.sym("quat_z")
        # velocity
        vel_x, vel_y, vel_z = ca.SX.sym("vel_x"), ca.SX.sym("vel_y"), ca.SX.sym("vel_z")

        # concatenated vector (x)
        self._quad_state = ca.vertcat(pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x, vel_y, vel_z)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym("thrust"), ca.SX.sym("wx"), ca.SX.sym("wy"), ca.SX.sym("wz")

        # concatenated vector (u)
        self._quad_action = ca.vertcat(thrust, wx, wy, wz)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        quad_state_dot = ca.vertcat(
            vel_x,
            vel_y,
            vel_z,
            0.5 * (-wx * quat_x - wy * quat_y - wz * quat_z),
            0.5 * (wx * quat_w + wz * quat_y - wy * quat_z),
            0.5 * (wy * quat_w - wz * quat_x + wx * quat_z),
            0.5 * (wz * quat_w + wy * quat_x - wx * quat_y),
            2 * (quat_w * quat_y + quat_x * quat_z) * (thrust / self._mass),
            2 * (quat_y * quat_z - quat_w * quat_x) * (thrust / self._mass),
            (quat_w * quat_w - quat_x * quat_x - quat_y * quat_y + quat_z * quat_z) * (thrust / self._mass) - self._gravity
        )

        self.quad_dynamics = ca.Function(
            "f",
            [self._quad_state, self._quad_action], [quad_state_dot],
            ["x", "u"], ["ode"]
        )

        # # Fold
        # basically self.f defines the system dynamics (going from input state and command to the derivative of the
        # state variables) and quad_dyn_int defines a function "rolling" this forward for the time step self._dt
        quad_dyn_int = self.quad_dynamics_integration(self._pred_time_step)
        quad_dyn_int_parallel = quad_dyn_int.map(self._num_pred_steps, "openmp")  # parallel

        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #

        # placeholder for the quadratic cost function
        # these are the three different cost functions:
        # - difference for the final state (i.e. only the state/position for the last time step)
        # - difference between state and middle of the gate (pendulum tracking error)
        # - regularisation penalty for the control input (compared to a hovering, i.e. only counter-acting gravity)
        delta_goal = ca.SX.sym("delta_goal", self._state_dim)
        delta_traj = ca.SX.sym("delta_traj", self._state_dim)
        delta_norm = ca.SX.sym("delta_norm", 4)
        delta_action = ca.SX.sym("delta_action", self._action_dim)

        # these are cost matrices for optimising with the MPC (probably don't need to cost_gap for the pendulum)
        cost_goal = delta_goal.T @ self._cost_matrix_goal @ delta_goal
        cost_traj = delta_traj.T @ self._cost_matrix_traj @ delta_traj
        cost_norm = ((ca.norm_2(delta_norm) - 1) ** 2) * self._cost_quaternion_norm
        cost_action = delta_action.T @ self._cost_matrix_action @ delta_action

        cost_goal = ca.Function("cost_goal", [delta_goal], [cost_goal])
        cost_traj = ca.Function("cost_traj", [delta_traj], [cost_traj])
        cost_norm = ca.Function("cost_norm", [delta_norm], [cost_norm])
        cost_action = ca.Function("cost_action", [delta_action], [cost_action])

        # all of the above is just used to define input-output relationships and the cost functions it seems
        # the actual NLP (with the objective, input/constraint bounds, initial guess) is defined below
        # # # # # # # # # # # # # # # # # # # #
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #

        self.variables = []  # nlp variables nlp_x
        self.variables_init_guess = []  # initial guess of nlp variables
        self.variables_lower_bound = []  # lower bound of the variables, lbw <= nlp_x
        self.variables_upper_bound = []  # upper bound of the variables, nlp_x <= ubw

        self.objective = 0  # objective
        self.constraints = []  # constraint functions g
        self.constraints_lower_bound = []  # lower bound of constraint functions, lbg < g
        self.constraints_upper_bound = []  # upper bound of constraint functions, g < ubg

        # concrete bounds
        action_min = [self._thrust_min, -self._w_max_xy, -self._w_max_xy, -self._w_max_yaw]
        action_max = [self._thrust_max, self._w_max_xy, self._w_max_xy, self._w_max_yaw]

        state_bound = ca.inf
        state_min = [-state_bound for _ in range(self._state_dim)]
        state_max = [+state_bound for _ in range(self._state_dim)]
        # state_min = ([-state_bound] * 3) + ([-1.1] * 4) + ([-state_bound] * 3)
        # state_max = ([state_bound] * 3) + ([1.1] * 4) + ([state_bound] * 3)

        constraint_min = [0 for _ in range(self._state_dim)]
        constraint_max = [0 for _ in range(self._state_dim)]

        # I guess this traj_input is just for the pendulum positions? but what is the solver actually doing with them then?
        # is the point that at a certain time, the quadrotor should be at the same position as the pendulum?
        # => in that case following a pre-defined trajectory would just be a matter of giving the place where
        #    the quadrotor should be at a certain time, right? TODO
        traj_input = ca.SX.sym("traj_input", self._state_dim * (self._num_pred_steps + 2))
        # ^ this is what? a vector of the initial and final quadrotor state with
        #   the intermediate quadrotor + pendulum states between them?
        #   => might be just the way to "insert" the actual positions/states into the program?
        traj_states = ca.SX.sym("traj_states", self._state_dim, self._num_pred_steps + 1)  # just the state for all time steps + the first/final (?) one?
        traj_actions = ca.SX.sym("traj_actions", self._action_dim, self._num_pred_steps)  # just the input command for all time steps (^ probably first then)

        traj_states_next = quad_dyn_int_parallel(traj_states[:, :self._num_pred_steps], traj_actions)
        # ^ this just seems to be parallelisation of the forward dynamics, over self._N

        # "Lift" initial conditions
        self.variables += [traj_states[:, 0]]  # add the initial state to the NLP variables
        self.variables_init_guess += self._quad_state_0  # add the values for the initial state to the guess for the above
        self.variables_lower_bound += state_min
        self.variables_upper_bound += state_max

        # # starting point.
        self.constraints += [traj_states[:, 0] - traj_input[:self._state_dim]]  # value of the constraint function (state
        self.constraints_lower_bound += constraint_min  # the difference above should be 0 at the start
        self.constraints_upper_bound += constraint_max

        # by the way self._N is defined, this means we optimise over the entire trajectory we want to fly
        # => this is not really something we can afford to do for a longer trajectory, so one would have to
        #    define some sort of time horizon and e.g. update the "goal position(s)" along the trajectory...
        for k in range(self._num_pred_steps):
            self.variables += [traj_actions[:, k]]  # add the commands to the variables
            self.variables_init_guess += self._quad_action_0  # for every time step the initial guess for the commands is hovering
            self.variables_lower_bound += action_min  # constrain the commands
            self.variables_upper_bound += action_max  # constrain the commands

            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = traj_input[ idx_k : idx_k_end]

            # cost for tracking the specified/given trajectory
            delta_traj_k = (traj_states[:, k + 1] - traj_input[(self._state_dim * (k + 1)):(self._state_dim * (k + 2))])
            cost_traj_k = cost_traj(delta_traj_k)

            # cost for deviating from the 1 norm
            # delta_norm_k = traj_states[3:7, k + 1]
            # cost_norm_k = cost_norm(delta_norm_k)
            # cost_norm_k = cost_norm(delta_norm_k)

            # just the regularisation cost for the control command (try to keep it close to hovering)
            delta_action_k = traj_actions[:, k] - [self._gravity, 0.0, 0.0, 0.0]
            cost_action_k = cost_action(delta_action_k)

            # self.objective = self.objective + cost_traj_k + cost_norm_k + cost_action_k
            self.objective = self.objective + cost_traj_k + cost_action_k

            # New NLP variable for state at end of interval <= after applying the control command I guess
            self.variables += [traj_states[:, k + 1]]
            self.variables_init_guess += self._quad_state_0
            self.variables_lower_bound += state_min
            self.variables_upper_bound += state_max

            # Add equality constraint
            # I guess this basically just means that when the program optimises over the state and control variables,
            # it needs to ensure that the state (traj_states) is consistent with the dynamics of the system (which are defined
            # by traj_states_next, the output of quad_dyn_int/quad_dyn_int_parallel)
            self.constraints += [traj_states_next[:, k] - traj_states[:, k + 1]]
            self.constraints_lower_bound += constraint_min
            self.constraints_upper_bound += constraint_max

            # self.constraints += [ca.norm_2(traj_states[:, k + 1]) - 1]
            # self.constraints_lower_bound += [0]
            # self.constraints_upper_bound += [0]

        # # # # # # # # # # # # # # # # # # #
        # -- ipopt
        # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            "verbose": False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }

        nlp_dict = {
            "f": self.objective,
            "x": ca.vertcat(*self.variables),
            "p": traj_input,
            "g": ca.vertcat(*self.constraints)
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

        """
        # jit (just-in-time compilation)
        print("Generating shared library........")
        cname = self.solver.generate_dependencies("mpc_v2.c")
        print(cname)
        import os
        os.system("gcc -fPIC -shared -O3 " + cname + " -o " + self.so_path)  # -O3
        """

        # # reload compiled mpc
        # print(self.so_path)
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)

    def solve(self, reference_trajectory):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #

        solution = self.solver(
            x0=self.variables_init_guess,
            lbx=self.variables_lower_bound,
            ubx=self.variables_upper_bound,
            p=reference_trajectory,  # => I guess this is the mysterious P from _initDynamics?
            lbg=self.constraints_lower_bound,
            ubg=self.constraints_upper_bound
        )

        cost = solution["f"]
        # exit()

        solution = solution["x"].full()
        optimal_action = solution[self._state_dim:(self._state_dim + self._action_dim)]
        # optimal_action = solution[:self._action_dim]

        # Warm initialization
        self.variables_init_guess = list(solution[(self._state_dim + self._action_dim)
                                                  :(2 * (self._state_dim + self._action_dim))])
        self.variables_init_guess += list(solution[self._state_dim + self._action_dim:])
        # self.variables_init_guess = list(solution[self._action_dim:(2 * self._action_dim)])
        # self.variables_init_guess += list(solution[self._action_dim:])
        # => improves the initial guess with the already computed commands I guess?
        # => I think this just repeats the "second" state/action combination because it skips the first one?

        predicted_traj = np.reshape(solution[:-self._state_dim], newshape=(-1, self._state_dim + self._action_dim))
        # predicted_traj = None

        # return optimal action, and a sequence of predicted optimal trajectory.
        return optimal_action.squeeze(), predicted_traj, cost

    def quad_dynamics_integration(self, pred_time_step):
        refine_steps = 4
        refine_dt = pred_time_step / refine_steps

        state_init = ca.SX.sym("state", self._state_dim)
        action = ca.SX.sym("action", self._action_dim)

        state = state_init
        for _ in range(refine_steps):
            # --------- RK4 ----------
            k1 = refine_dt * self.quad_dynamics(state, action)
            k2 = refine_dt * self.quad_dynamics(state + 0.5 * k1, action)
            k3 = refine_dt * self.quad_dynamics(state + 0.5 * k2, action)
            k4 = refine_dt * self.quad_dynamics(state + k3, action)

            state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        quad_dyn_int = ca.Function("quad_dyn_int", [state_init, action], [state])
        return quad_dyn_int
