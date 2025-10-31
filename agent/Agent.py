from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        self.goalie_can_take_ball = False
        self.goalie_last_holder_unum = None

        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation

        # Kickoff state: prevent double-touch by the kicker until someone else touches the ball
        # kicker_unum is set when we initiate the kickoff. lock_active becomes True once the ball leaves the kickoff spot
        # and remains True until a teammate (not the kicker) or an opponent is detected as touching/nearest to the ball.
        self.kickoff_kicker_unum = None
        self.kickoff_lock_active = False
        self.last_restart_ball_pos = None
        self.last_restart_mode = None


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            # Reset kickoff guard state on game over
            self.kickoff_kicker_unum = None
            self.kickoff_lock_active = False
            self.last_restart_ball_pos = None
            self.last_restart_mode = None
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            # Reset kickoff guard state when beaming
            self.kickoff_kicker_unum = None
            self.kickoff_lock_active = False
            self.last_restart_ball_pos = None
            self.last_restart_mode = None
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            # Reset kickoff guard state when beaming
            self.kickoff_kicker_unum = None
            self.kickoff_lock_active = False
            self.last_restart_ball_pos = None
            self.last_restart_mode = None
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                # Before kickoff: ensure no stale lock
                self.kickoff_kicker_unum = None
                self.kickoff_lock_active = False
                self.last_restart_ball_pos = None
                self.last_restart_mode = None
                pass


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""



        



    def select_skill(self,strategyData):
        #--------------------------------------- 2. Decide action
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options

        #------------------------------------------------------
        # PlayOn: Closest agent runs to ball, others spread out. Once it has the ball it runs with it to the opponent's goal and passes when opponent gets close.
        if strategyData.play_mode == self.world.M_PLAY_ON:
            return self._play_on_strategy(strategyData)

        #------------------------------------------------------
        # Their kickoff: barrier around center while ball is at origin; else switch to PlayOn behavior
        if strategyData.play_mode == self.world.M_THEIR_KICKOFF:
            ball_2d = self.world.ball_abs_pos[:2]
            center = np.array((0.0, 0.0))
            if np.linalg.norm(ball_2d - center) < 0.2:  # ball still on kickoff spot
                drawer.annotation((0,10.5), "Their Kickoff: Huddle", drawer.Color.yellow, "status")

                # Barrier circle parameters
                radius = 2.7
                n = max(1, len(strategyData.teammate_positions))

                # Arc on our half (x <= 0), centered at 180°, span up to 120°
                max_span_deg = 120.0
                span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
                start_deg = 145.0 + span_deg / 2.0
                end_deg = 145.0 - span_deg / 2.0
                angles_deg = np.linspace(start_deg, end_deg, n)

                arc_points = []
                for ang in angles_deg:
                    a = np.deg2rad(ang)
                    p = center + radius * np.array((np.cos(a), np.sin(a)))
                    if p[0] > -0.1:  # ensure on our half
                        p[0] = -0.1
                    arc_points.append(tuple(p))

                my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
                my_target = arc_points[my_idx]

                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(center)
                drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "kickoff huddle")
                return self.move(my_target, orientation=desired_ori)
            else:
                # Ball has been kicked: behave like PlayOn
                return self._play_on_strategy(strategyData)

        #------------------------------------------------------
        # Our kickoff: assign closest player as kicker; others hold formation. Kicker does a direct long kick to right field.
        if strategyData.play_mode == self.world.M_OUR_KICKOFF:
            goal_right = (15, 0)
            r = self.world.robot
            mypos_2d = r.loc_head_position[:2]
            ball_2d = self.world.ball_abs_pos[:2]
            center = np.array((0.0, 0.0))

            # Ball is still on the kickoff spot: select kicker and execute the first kick
            kicker_unum = strategyData.active_player_unum  # closest to the ball at kickoff
            # Remember who the kicker is for the lock once the ball moves
            if self.kickoff_kicker_unum is None:
                self.kickoff_kicker_unum = kicker_unum
                self.last_restart_ball_pos = np.array(ball_2d)
                self.last_restart_mode = strategyData.play_mode

            # If the ball has left the kickoff spot, activate the double-touch lock and behave like PlayOn
            if (
                self.kickoff_kicker_unum is not None and not self.kickoff_lock_active
                and self.last_restart_ball_pos is not None
                and np.linalg.norm(np.array(ball_2d) - self.last_restart_ball_pos) > 0.25
            ):
                self.kickoff_lock_active = True
                return self._play_on_strategy(strategyData)

            if strategyData.robot_model.unum == kicker_unum:
                # Visualize intent
                drawer.annotation((0,10.5), "Our Kickoff: Kicker", drawer.Color.yellow, "status")
                drawer.line(tuple(mypos_2d), goal_right, 2, drawer.Color.red, "kickoff kick")
                # Perform the initial kick (we do NOT activate the lock yet; it triggers when the ball actually moves)
                return self.kick_through_best_gap(strategyData, mypos_2d, ball_2d, goal_right)
            else:
                # Hold a simple formation while waiting for the kick
                drawer.annotation((0,10.5), "Our Kickoff: Formation", drawer.Color.cyan, "status")
                formation_positions = GenerateBasicFormation()
                point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
                strategyData.my_desired_position = point_preferences[strategyData.player_unum]
                strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                    strategyData.my_desired_position)
                drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "kickoff setup")
                return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

        #------------------------------------------------------
        # Their kick-in: create barrier around the ball so our agents stay at least 2.7m away
        if strategyData.play_mode == self.world.M_THEIR_KICK_IN:
            ball_2d = self.world.ball_abs_pos[:2].copy()
            drawer.annotation((0,10.5), "Their Kick-In: Barrier", drawer.Color.yellow, "status")

            radius = 2.7
            n = max(1, len(strategyData.teammate_positions))

            # place arc facing our goal (defend)
            our_goal = (-15.0, 0.0)
            dir_to_goal = np.degrees(np.arctan2(our_goal[1]-ball_2d[1], our_goal[0]-ball_2d[0]))
            max_span_deg = 120.0
            span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
            start_deg = dir_to_goal - 35 + span_deg/2.0
            end_deg = dir_to_goal - 35 - span_deg/2.0
            angles_deg = np.linspace(start_deg, end_deg, n)

            arc_points = []
            for ang in angles_deg:
                a = np.deg2rad(ang)
                p = ball_2d + radius * np.array((np.cos(a), np.sin(a)))
                # clamp inside field bounds (optional)
                p[0] = np.clip(p[0], -15.0, 15.0)
                p[1] = np.clip(p[1], -10.0, 10.0)
                arc_points.append(tuple(p))

            my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
            my_target = arc_points[my_idx]

            desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
            drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "their kickin barrier")
            return self.move(my_target, orientation=desired_ori)

        #------------------------------------------------------
        # Our kick-in: assign closest player as kicker, others form arc around the ball
        if strategyData.play_mode == self.world.M_OUR_KICK_IN:
            ball_2d = self.world.ball_abs_pos[:2]
            goal_right = (15, 0)
            r = self.world.robot
            mypos_2d = r.loc_head_position[:2]

            # Find the kicker (closest teammate to the ball)
            teammate_positions = strategyData.teammate_positions
            dists = [np.linalg.norm(np.array(pos) - ball_2d) if pos is not None else float('inf') for pos in teammate_positions]
            kicker_idx = int(np.argmin(dists))
            kicker_unum = kicker_idx + 1

            # Remember who the kicker is for the lock once the ball moves
            if self.kickoff_kicker_unum is None:
                self.kickoff_kicker_unum = kicker_unum
                self.last_restart_ball_pos = np.array(ball_2d)
                self.last_restart_mode = strategyData.play_mode

            # Activate the double-touch lock if the ball has moved from the restart spot
            if (
                self.kickoff_kicker_unum is not None and not self.kickoff_lock_active
                and self.last_restart_ball_pos is not None
                and np.linalg.norm(np.array(ball_2d) - self.last_restart_ball_pos) > 0.25
            ):
                self.kickoff_lock_active = True

            if strategyData.robot_model.unum == kicker_unum:
                if self.kickoff_lock_active:
                    # If lock is active, kicker must avoid the ball
                    to_me = mypos_2d - ball_2d
                    d = np.linalg.norm(to_me)
                    if d < 0.6:
                        if d < 1e-3:
                            to_me = np.array([-1.0, 0.0])
                        else:
                            to_me = to_me / d
                        safe_point = mypos_2d + to_me * 0.8
                        desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                        drawer.line(tuple(mypos_2d), tuple(safe_point), 2, drawer.Color.red, "avoid ball")
                        return self.move(tuple(safe_point), orientation=desired_ori)
                else:
                    # Kicker: go to ball and kick toward goal
                    drawer.annotation((0,10.5), "Our Kick-In: Kicker", drawer.Color.yellow, "status")
                    drawer.line(tuple(mypos_2d), goal_right, 2, drawer.Color.red, "kickin kick")
                    return self.kick_through_best_gap(strategyData, mypos_2d, ball_2d, goal_right)
            else:
                # Support: form arc around the ball, facing the goal
                drawer.annotation((0,10.5), "Our Kick-In: Support", drawer.Color.cyan, "status")
                n = max(1, len(teammate_positions))
                radius = 2.7
                goal_dir = np.degrees(np.arctan2(goal_right[1] - ball_2d[1], goal_right[0] - ball_2d[0]))
                max_span_deg = 120.0
                span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
                start_deg = goal_dir - 35 + span_deg/2.0
                end_deg = goal_dir - 35 - span_deg/2.0
                angles_deg = np.linspace(start_deg, end_deg, n)

                arc_points = []
                for ang in angles_deg:
                    a = np.deg2rad(ang)
                    p = ball_2d + radius * np.array((np.cos(a), np.sin(a)))
                    # clamp inside field bounds (optional)
                    p[0] = np.clip(p[0], -15.0, 15.0)
                    p[1] = np.clip(p[1], -10.0, 10.0)
                    arc_points.append(tuple(p))

                my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
                my_target = arc_points[my_idx]
                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(goal_right)
                drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "our kickin support")
                return self.move(my_target, orientation=desired_ori)

        #------------------------------------------------------
        # Their free kick: create barrier around the ball so our agents stay at least 2.7m away
        if strategyData.play_mode == self.world.M_THEIR_FREE_KICK:
            ball_2d = self.world.ball_abs_pos[:2].copy()
            drawer.annotation((0,10.5), "Their Kick-In: Barrier", drawer.Color.yellow, "status")

            radius = 2.7
            n = max(1, len(strategyData.teammate_positions))

            # place arc facing our goal (defend)
            our_goal = (-15.0, 0.0)
            dir_to_goal = np.degrees(np.arctan2(our_goal[1]-ball_2d[1], our_goal[0]-ball_2d[0]))
            max_span_deg = 120.0
            span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
            start_deg = dir_to_goal - 35 + span_deg/2.0
            end_deg = dir_to_goal - 35 - span_deg/2.0
            angles_deg = np.linspace(start_deg, end_deg, n)

            arc_points = []
            for ang in angles_deg:
                a = np.deg2rad(ang)
                p = ball_2d + radius * np.array((np.cos(a), np.sin(a)))
                # clamp inside field bounds (optional)
                p[0] = np.clip(p[0], -15.0, 15.0)
                p[1] = np.clip(p[1], -10.0, 10.0)
                arc_points.append(tuple(p))

            my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
            my_target = arc_points[my_idx]

            desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
            drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "their kickin barrier")
            return self.move(my_target, orientation=desired_ori)

        #------------------------------------------------------
        # Our free kick: assign closest player as kicker, others form arc around the ball
        if strategyData.play_mode == self.world.M_OUR_FREE_KICK:
            ball_2d = self.world.ball_abs_pos[:2]
            goal_right = (15, 0)
            r = self.world.robot
            mypos_2d = r.loc_head_position[:2]

            # Find the kicker (closest teammate to the ball)
            teammate_positions = strategyData.teammate_positions
            dists = [np.linalg.norm(np.array(pos) - ball_2d) if pos is not None else float('inf') for pos in teammate_positions]
            kicker_idx = int(np.argmin(dists))
            kicker_unum = kicker_idx + 1

            # Remember who the kicker is for the lock once the ball moves
            if self.kickoff_kicker_unum is None:
                self.kickoff_kicker_unum = kicker_unum
                self.last_restart_ball_pos = np.array(ball_2d)
                self.last_restart_mode = strategyData.play_mode

            # Activate the double-touch lock if the ball has moved from the restart spot
            if (
                self.kickoff_kicker_unum is not None and not self.kickoff_lock_active
                and self.last_restart_ball_pos is not None
                and np.linalg.norm(np.array(ball_2d) - self.last_restart_ball_pos) > 0.25
            ):
                self.kickoff_lock_active = True

            if strategyData.robot_model.unum == kicker_unum:
                if self.kickoff_lock_active:
                    # If lock is active, kicker must avoid the ball
                    to_me = mypos_2d - ball_2d
                    d = np.linalg.norm(to_me)
                    if d < 0.6:
                        if d < 1e-3:
                            to_me = np.array([-1.0, 0.0])
                        else:
                            to_me = to_me / d
                        safe_point = mypos_2d + to_me * 0.8
                        desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                        drawer.line(tuple(mypos_2d), tuple(safe_point), 2, drawer.Color.red, "avoid ball")
                        return self.move(tuple(safe_point), orientation=desired_ori)
                else:
                    # Kicker: go to ball and kick toward goal
                    drawer.annotation((0,10.5), "Our Kick-In: Kicker", drawer.Color.yellow, "status")
                    drawer.line(tuple(mypos_2d), goal_right, 2, drawer.Color.red, "kickin kick")
                    return self.kick_through_best_gap(strategyData, mypos_2d, ball_2d, goal_right)
            else:
                # Support: form arc around the ball, facing the goal
                drawer.annotation((0,10.5), "Our Kick-In: Support", drawer.Color.cyan, "status")
                n = max(1, len(teammate_positions))
                radius = 2.7
                goal_dir = np.degrees(np.arctan2(goal_right[1] - ball_2d[1], goal_right[0] - ball_2d[0]))
                max_span_deg = 120.0
                span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
                start_deg = goal_dir - 35 + span_deg/2.0
                end_deg = goal_dir - 35 - span_deg/2.0
                angles_deg = np.linspace(start_deg, end_deg, n)

                arc_points = []
                for ang in angles_deg:
                    a = np.deg2rad(ang)
                    p = ball_2d + radius * np.array((np.cos(a), np.sin(a)))
                    # clamp inside field bounds (optional)
                    p[0] = np.clip(p[0], -15.0, 15.0)
                    p[1] = np.clip(p[1], -10.0, 10.0)
                    arc_points.append(tuple(p))

                my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
                my_target = arc_points[my_idx]
                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(goal_right)
                drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "our kickin support")
                return self.move(my_target, orientation=desired_ori)

        #------------------------------------------------------
        # Their goal kick: make barrier with closest player at centre
        if strategyData.play_mode == self.world.M_THEIR_GOAL_KICK:
            ball_2d = self.world.ball_abs_pos[:2].copy()
            drawer.annotation((0,10.5), "Their Goal Kick: Barrier", drawer.Color.yellow, "status")

            radius = 2.7
            n = max(1, len(strategyData.teammate_positions))

            # Find the closest teammate to the ball (to be placed in the center of the arc)
            teammate_positions = strategyData.teammate_positions
            dists = [np.linalg.norm(np.array(pos) - ball_2d) if pos is not None else float('inf') for pos in teammate_positions]
            sorted_indices = np.argsort(dists)
            # The closest player will be in the center of the arc
            center_idx = sorted_indices[0]

            # Arc parameters: face the field (our goal is at -15,0, so arc faces +x)
            arc_center_angle = 0  # 0 degrees (facing +x, i.e., toward the field)
            max_span_deg = 120.0
            span_deg = min(max_span_deg, max(0.0, 20.0 * (n - 1)))
            # Center the arc on arc_center_angle, with the closest player in the middle
            angles_deg = np.linspace(arc_center_angle + span_deg/2.0, arc_center_angle - span_deg/2.0, n)

            # Assign arc points so that the closest player gets the center angle
            arc_points = []
            for ang in angles_deg:
                a = np.deg2rad(ang)
                p = ball_2d + radius * np.array((np.cos(a), np.sin(a)))
                p[0] = np.clip(p[0], -15.0, 15.0)
                p[1] = np.clip(p[1], -10.0, 10.0)
                arc_points.append(tuple(p))

            # Map teammates to arc points: closest player gets center, others fill outwards
            arc_assignment = [None] * n
            mid = n // 2
            arc_assignment[center_idx] = mid
            left = mid - 1
            right = mid + 1
            for i, idx in enumerate(sorted_indices[1:]):
                if i % 2 == 0:
                    if right < n:
                        arc_assignment[idx] = right
                        right += 1
                else:
                    if left >= 0:
                        arc_assignment[idx] = left
                        left -= 1

            my_idx = min(max(0, strategyData.player_unum - 1), n - 1)
            my_arc_idx = arc_assignment[my_idx]
            my_target = arc_points[my_arc_idx]

            desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
            drawer.line(strategyData.mypos, my_target, 2, drawer.Color.blue, "their goal kick barrier")
            return self.move(my_target, orientation=desired_ori)

        #------------------------------------------------------
        # Our goal kick: only the goalie moves, others hold formation
        if strategyData.play_mode == self.world.M_OUR_GOAL_KICK:
            ball_2d = self.world.ball_abs_pos[:2]
            goal_right = (15, 0)
            r = self.world.robot
            mypos_2d = r.loc_head_position[:2]

            # Only the goalie (usually unum == 1) takes the goal kick
            if strategyData.robot_model.unum == 1:
                # Find closest opponent to the ball
                opponents = [np.array(pos) for pos in strategyData.opponent_positions if pos is not None]
                if opponents:
                    dists = [np.linalg.norm(opp - ball_2d) for opp in opponents]
                    closest_opp = opponents[int(np.argmin(dists))]
                    rel = closest_opp - ball_2d
                    opp_angle = np.arctan2(rel[1], rel[0])

                    # Candidate directions: 1 radian left/right of opponent
                    angle_offset = np.deg2rad(30)
                    left_angle = opp_angle + angle_offset
                    right_angle = opp_angle - angle_offset

                    # Direction to goal
                    goal_angle = np.arctan2(goal_right[1] - ball_2d[1], goal_right[0] - ball_2d[0])

                    # Compute angle difference to goal for both candidates
                    left_diff = abs(np.arctan2(np.sin(left_angle - goal_angle), np.cos(left_angle - goal_angle)))
                    right_diff = abs(np.arctan2(np.sin(right_angle - goal_angle), np.cos(right_angle - goal_angle)))

                    # Pick the direction closer to the goal
                    best_angle = left_angle if left_diff < right_diff else right_angle
                    target = ball_2d + 10.0 * np.array([np.cos(best_angle), np.sin(best_angle)])

                    drawer = self.world.draw
                    drawer.annotation((0,10.5), "Our Goal Kick: Goalie", drawer.Color.yellow, "status")
                    drawer.line(tuple(mypos_2d), tuple(target), 2, drawer.Color.red, "goal kick")
                    return self.kickTarget(strategyData, mypos_2d, target)
                else:
                    # No opponents: kick straight to goal
                    return self.kickTarget(strategyData, mypos_2d, goal_right)
            else:
                # Other players: move to support positions (reuse formation or arc logic as desired)
                formation_positions = GenerateBasicFormation()
                point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
                strategyData.my_desired_position = point_preferences[strategyData.player_unum]
                strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                    strategyData.my_desired_position)
                drawer = self.world.draw
                drawer.annotation((0,10.5), "Our Goal Kick: Support", drawer.Color.cyan, "status")
                drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "goal kick support")
                return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

        #------------------------------------------------------
        # Default: no predefined behavior -> behave like PlayOn
        return self._play_on_strategy(strategyData)
        

    def _play_on_strategy(self, strategyData):
        drawer = self.world.draw
        goal = (15, 0)  # opponent's goal
        r = self.world.robot
        ball_2d = self.world.ball_abs_pos[:2]
        mypos_2d = r.loc_head_position[:2]

        # --- Goalie logic (agent 1) ---
        if strategyData.robot_model.unum == 1:
            our_goal_center = np.array([-15.0, 0.0])
            ball_vec = ball_2d - our_goal_center
            ball_dist = np.linalg.norm(ball_vec)
            ball_dir = ball_vec / (ball_dist + 1e-6)

            # Only consider taking the ball if it's within 3m of the goal
            if ball_dist < 3.0:
                # Find the closest agent (not the goalie) to the ball
                min_dist = float('inf')
                closest_holder_unum = None
                closest_holder_is_teammate = False
                # Teammates (excluding self)
                for idx, pos in enumerate(strategyData.teammate_positions):
                    if pos is not None and len(pos) >= 2 and not np.allclose(pos[:2], mypos_2d):
                        dist = np.linalg.norm(np.array(pos[:2]) - ball_2d)
                        if dist < min_dist and dist < 0.35:  # possession threshold
                            min_dist = dist
                            closest_holder_unum = idx + 1
                            closest_holder_is_teammate = True
                # Opponents
                for idx, pos in enumerate(strategyData.opponent_positions):
                    if pos is not None and len(pos) >= 2:
                        dist = np.linalg.norm(np.array(pos[:2]) - ball_2d)
                        if dist < min_dist and dist < 0.35:
                            min_dist = dist
                            closest_holder_unum = -(idx + 1)  # negative for opponent
                            closest_holder_is_teammate = False

                # Check if goalie itself has the ball
                goalie_has_ball = np.linalg.norm(mypos_2d - ball_2d) < 0.28

                # Logic:
                # 1. If goalie_can_take_ball is False, wait/defend until closest_holder_unum loses the ball
                # 2. If closest_holder_unum is None (no one has the ball), set goalie_can_take_ball = True
                # 3. If goalie_can_take_ball is True, allow goalie to take and kick the ball
                # 4. If goalie loses the ball, reset goalie_can_take_ball to False

                if not self.goalie_can_take_ball:
                    # Track who is holding the ball
                    if closest_holder_unum is not None:
                        self.goalie_last_holder_unum = closest_holder_unum
                        # Wait/defend as usual
                        desired_dist = min(1.5, ball_dist - 0.3) if ball_dist > 0.3 else 0.3
                        target_pos = our_goal_center + ball_dir * desired_dist
                        drawer.annotation((0,10.5), "Goalie: Waiting for holder to lose ball", drawer.Color.yellow, "status")
                        drawer.line(tuple(mypos_2d), tuple(target_pos), 2, drawer.Color.blue, "goalie block")
                        desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                        return self.move(target_pos, orientation=desired_ori)
                    else:
                        # No one has the ball: allow goalie to take it
                        self.goalie_can_take_ball = True
                        self.goalie_last_holder_unum = None

                # If goalie_can_take_ball is True, try to take and kick the ball
                if self.goalie_can_take_ball:
                    if goalie_has_ball:
                        # Kick to goal
                        drawer.annotation((0,10.5), "Goalie: Kick to Goal", drawer.Color.yellow, "status")
                        return self.kickTarget(strategyData, mypos_2d, goal)
                    else:
                        # Move to ball to take possession
                        drawer.annotation((0,10.5), "Goalie: Go to Ball", drawer.Color.yellow, "status")
                        desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                        drawer.line(tuple(mypos_2d), tuple(ball_2d), 2, drawer.Color.green, "goalie to ball")
                        # If goalie is close but doesn't have the ball, keep trying
                        if np.linalg.norm(mypos_2d - ball_2d) < 0.35:
                            # If another agent gets the ball, go back to waiting
                            if closest_holder_unum is not None and not goalie_has_ball:
                                self.goalie_can_take_ball = False
                                self.goalie_last_holder_unum = closest_holder_unum
                                desired_dist = min(1.5, ball_dist - 0.3) if ball_dist > 0.3 else 0.3
                                target_pos = our_goal_center + ball_dir * desired_dist
                                drawer.annotation((0,10.5), "Goalie: Waiting for holder to lose ball", drawer.Color.yellow, "status")
                                drawer.line(tuple(mypos_2d), tuple(target_pos), 2, drawer.Color.blue, "goalie block")
                                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                                return self.move(target_pos, orientation=desired_ori)
                        return self.move(ball_2d, orientation=desired_ori)

                # If goalie had the ball but now lost it, reset flag
                if self.goalie_can_take_ball and not goalie_has_ball:
                    self.goalie_can_take_ball = False

                # Default: wait/defend
                desired_dist = min(1.5, ball_dist - 0.3) if ball_dist > 0.3 else 0.3
                target_pos = our_goal_center + ball_dir * desired_dist
                drawer.annotation((0,10.5), "Goalie: Block", drawer.Color.yellow, "status")
                drawer.line(tuple(mypos_2d), tuple(target_pos), 2, drawer.Color.blue, "goalie block")
                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                return self.move(target_pos, orientation=desired_ori)

            # If the ball is not within 3m of the goal, always wait/defend
            desired_dist = min(1.5, ball_dist - 0.3) if ball_dist > 0.3 else 0.3
            target_pos = our_goal_center + ball_dir * desired_dist
            drawer.annotation((0,10.5), "Goalie: Block (ball too far)", drawer.Color.yellow, "status")
            drawer.line(tuple(mypos_2d), tuple(target_pos), 2, drawer.Color.blue, "goalie block")
            desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
            return self.move(target_pos, orientation=desired_ori)
        
        # If a kickoff just happened and we already know the kicker, ensure the lock is active
        if self.kickoff_kicker_unum is not None and not self.kickoff_lock_active:
            self.kickoff_lock_active = True

        # Define possession and pressure thresholds
        has_ball = np.linalg.norm(ball_2d - mypos_2d) < 0.28
        opponent_close = (getattr(strategyData, "min_opponent_ball_dist", None) is not None 
                          and strategyData.min_opponent_ball_dist < 1.2)

        # Decide active player: the closest to the ball, with optional kickoff double-touch override
        effective_active_unum = strategyData.active_player_unum
        if (self.kickoff_kicker_unum is not None and self.kickoff_lock_active
            and effective_active_unum == self.kickoff_kicker_unum):
            # Choose nearest non-kicker as temporary active player
            non_kicker_indices = [i for i in range(len(strategyData.teammate_positions)) if (i+1) != self.kickoff_kicker_unum]
            # filter out teammates with unknown positions
            valid_indices = [i for i in non_kicker_indices if (strategyData.teammate_positions[i] is not None and len(strategyData.teammate_positions[i]) >= 2)]
            if valid_indices:
                dists = [np.linalg.norm(np.array(strategyData.teammate_positions[i][:2]) - ball_2d) for i in valid_indices]
                min_idx = valid_indices[int(np.argmin(dists))]
                effective_active_unum = min_idx + 1
            else:
                # Hard fallback: pick the lowest-numbered teammate that isn't the kicker
                candidate_unums = [i+1 for i in non_kicker_indices]
                if candidate_unums:
                    effective_active_unum = candidate_unums[0]

        # If a non-kicker already has possession, clear the restriction
        if self.kickoff_kicker_unum is not None and self.kickoff_lock_active:
            # Estimate possession by checking nearest teammate to ball is not the kicker and is close
            all_indices = [i for i in range(len(strategyData.teammate_positions)) if (strategyData.teammate_positions[i] is not None and len(strategyData.teammate_positions[i]) >= 2)]
            if all_indices:
                tdists = [np.linalg.norm(np.array(strategyData.teammate_positions[i][:2]) - ball_2d) for i in all_indices]
                nn_i = int(np.argmin(tdists))
                nearest_unum = all_indices[nn_i] + 1
                if nearest_unum != self.kickoff_kicker_unum and tdists[nn_i] < 0.35:
                    # Another teammate touched the ball: unlock and forget the kicker
                    self.kickoff_lock_active = False
                    self.kickoff_kicker_unum = None
            # Or if an opponent clearly touched the ball
            if getattr(strategyData, "min_opponent_ball_dist", None) is not None and strategyData.min_opponent_ball_dist < 0.35:
                self.kickoff_lock_active = False
                self.kickoff_kicker_unum = None

        i_am_active = (effective_active_unum == strategyData.robot_model.unum)

        if i_am_active:
            drawer.annotation((0,10.5), "PlayOn: Active - Ball Phase", drawer.Color.yellow, "status")
            # If I don't have the ball yet, go get it (aim body towards the ball)
            if not has_ball:
                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                drawer.line(tuple(mypos_2d), tuple(ball_2d), 2, drawer.Color.green, "to ball")
                return self.move(ball_2d, orientation=desired_ori)

            # I have the ball: either dribble to goal or pass if pressured
            # If close enough to goal, take the shot
            dist_to_goal = np.linalg.norm(np.array(goal) - mypos_2d)
            SHOOT_DISTANCE = 3.5  # meters; conservative to ensure stable shot
            if dist_to_goal < SHOOT_DISTANCE:
                drawer.line(tuple(mypos_2d), goal, 3, drawer.Color.orange, "dribble into goal")
                # Dribble into the goal instead of kicking
                desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(goal)
                return self.behavior.execute("Dribble", desired_ori, True, 1.0, False)

            # Default when I have the ball: prioritize stable, continuous dribbling toward goal.
            # Reduce pass-by-pressure aggressiveness: only pass if opponent is very close AND a clearly better teammate exists.
            desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(goal)
            # Slightly stricter threshold to avoid giving up dribble too early
            OPPONENT_PRESSURE_THRESHOLD = 0.9
            if opponent_close and getattr(strategyData, "min_opponent_ball_dist", 999) < OPPONENT_PRESSURE_THRESHOLD:
                # existing logic: look for a pass candidate (nearest teammate, excluding self)
                my_unum = strategyData.player_unum
                nearest_teammate_pos = None
                nearest_dist = float("inf")
                for idx, mate_pos in enumerate(strategyData.teammate_positions):
                    mate_unum = idx + 1
                    # skip self and unknown teammate positions
                    if mate_unum == my_unum or mate_pos is None or len(mate_pos) < 2:
                        continue
                    mate_pos_2d = np.array(mate_pos[:2])
                    d = np.linalg.norm(mate_pos_2d - mypos_2d)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_teammate_pos = mate_pos_2d

                # Only consider the pass if teammate is clearly open (not just marginally)
                if nearest_teammate_pos is not None and nearest_dist < 3.5:
                    drawer.line(tuple(mypos_2d), tuple(nearest_teammate_pos), 2, drawer.Color.red, "pass line")
                    return self.kickTarget(strategyData, tuple(mypos_2d), tuple(nearest_teammate_pos))

            # Otherwise, continue dribbling aggressively toward the goal.
            # Use Dribble behavior directly to avoid walking + replanning jitter.
            drawer.line(tuple(mypos_2d), goal, 2, drawer.Color.orange, "dribble line")
            return self.behavior.execute("Dribble", desired_ori, True, 1.0, False)
        else:
            # Not active: spread using role assignment so we are available for a pass
            drawer.annotation((0,10.5), "PlayOn: Support - Formation", drawer.Color.cyan, "status")
            # If I'm the kickoff kicker while the lock is active, explicitly avoid the ball
            if self.kickoff_kicker_unum == strategyData.robot_model.unum and self.kickoff_lock_active:
                # Step away from ball if too close
                to_me = mypos_2d - ball_2d
                d = np.linalg.norm(to_me)
                if d < 0.6:
                    if d < 1e-3:
                        to_me = np.array([-1.0, 0.0])  # arbitrary safe direction
                    else:
                        to_me = to_me / d
                    safe_point = mypos_2d + to_me * 0.8
                    desired_ori = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_2d)
                    drawer.line(tuple(mypos_2d), tuple(safe_point), 2, drawer.Color.red, "avoid ball")
                    return self.move(tuple(safe_point), orientation=desired_ori)
            formation_positions = GenerateBasicFormation()
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
            strategyData.my_desired_position = point_preferences[strategyData.player_unum]
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(
                strategyData.my_desired_position)
            drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desired_orientation)

    #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")

    def kick_through_best_gap(self, strategyData, mypos_2d, ball_2d, goal_2d, min_gap_width=0.6, kick_distance=5.0):
        """
        Kick the ball through the widest available gap towards the goal.
        """
        # Gather opponent positions
        opponents = [np.array(pos) for pos in strategyData.opponent_positions if pos is not None]
        if not opponents:
            # No opponents: kick straight to goal
            return self.kickTarget(strategyData, mypos_2d, goal_2d)

        # Compute angles to opponents from the ball
        angles = []
        for opp in opponents:
            rel = opp - ball_2d
            angle = np.arctan2(rel[1], rel[0])
            dist = np.linalg.norm(rel)
            angles.append((angle, dist, opp))
        angles.sort()

        # Add "virtual" opponents at -pi and +pi to close the field
        angles = [(-np.pi, 1000, None)] + angles + [(np.pi, 1000, None)]

        # Find all gaps between sorted opponent angles
        best_gap = None
        best_gap_angle = None
        goal_angle = np.arctan2(goal_2d[1] - ball_2d[1], goal_2d[0] - ball_2d[0])
        min_angle_diff = float('inf')

        for i in range(len(angles) - 1):
            a1, d1, _ = angles[i]
            a2, d2, _ = angles[i+1]
            # For each gap, check if the gap at 1.5m from the ball is wide enough
            arc1 = ball_2d + 1.5 * np.array([np.cos(a1), np.sin(a1)])
            arc2 = ball_2d + 1.5 * np.array([np.cos(a2), np.sin(a2)])
            gap_width = np.linalg.norm(arc2 - arc1)
            if gap_width >= min_gap_width:
                gap_center = (a1 + a2) / 2
                # Only consider gaps whose center is within ±90° of the goal direction
                center_diff = np.arctan2(np.sin(gap_center - goal_angle), np.cos(gap_center - goal_angle))
                if abs(center_diff) > np.pi / 2:
                    continue  # Skip gaps that are too far from the goal direction
                angle_diff = abs(center_diff)
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_gap = (a1, a2)
                    best_gap_angle = gap_center

        if best_gap is not None:
            # Kick through the center of the best gap
            target = ball_2d + kick_distance * np.array([np.cos(best_gap_angle), np.sin(best_gap_angle)])
            return self.kickTarget(strategyData, mypos_2d, target)
        else:
            # No gap found: fallback to goal
            return self.kickTarget(strategyData, mypos_2d, goal_2d)