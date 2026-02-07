%% MPC for Soil Water Balance with Precipitation
% Initial model with external disturbance
% Objective: Evaluate MPC reaction to rain events
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% Precipitation is added as an external disturbance.
% The system no longer depends solely on irrigation control; with rain, 
% irrigation decreases, achieving stability at lower costs. 
% Rain "helps" the system. Still independent of t and theta.

%% -------------------------
% System Parameters
% --------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]
ETc = 4.0;        % Evapotranspiration [mm/day]
D   = 0.5;        % Drainage [mm/day]
theta_ref = 0.25; % Target moisture (Set-point)

%% -------------------------
% MPC Horizon
% --------------------------
N = 10;           % Prediction horizon

%% -------------------------
% Symbolic Definition
% --------------------------
x = SX.sym('x');        % State: Soil moisture
u = SX.sym('u');        % Control: Irrigation
p = SX.sym('p');        % Precipitation (Parameter/Disturbance)

% Dynamic Model
x_next = x + dt*(1/(Zr*1000))*(u + p - ETc - D);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% Optimization Variables
% --------------------------
X = SX.sym('X',N+1);       % State trajectory
U = SX.sym('U',N);         % Control actions
X0 = SX.sym('X0');         % Initial state
P  = SX.sym('P',N);        % Precipitation profile over horizon

%% -------------------------
% Objective Function and Constraints
% --------------------------
J = 0;
g = [];

% Initial condition constraint
g = [g; X(1) - X0];

for k = 1:N
    % Cost Function (Tracking error + Control effort)
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    
    % Dynamics including precipitation disturbance
    g = [g; X(k+1) - f(X(k),U(k),P(k))];
end

%% -------------------------
% NLP Formulation
% --------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0; P]);
opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Bounds (Constraints)
% --------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    0.15*ones(N+1,1);   % Minimum theta
    0.0*ones(N,1)       % Minimum irrigation
];

ubw = [ ...
    0.35*ones(N+1,1);   % Maximum theta
    6.0*ones(N,1)       % Maximum irrigation
];

%% -------------------------
% Closed-loop Simulation
% --------------------------
Tsim = 30;             % Simulation days
xk = 0.18;             % Initial moisture
theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
p_hist     = zeros(Tsim,1);

% Rain Profile (Example disturbance)
P_sim = zeros(Tsim,1);
P_sim(10:12) = 6;      % Rain event [mm/day]

% Initialization
w0 = zeros((N+1)+N,1);

for t = 1:Tsim
    % Extract precipitation horizon
    if t+N-1 <= Tsim
        P_hor = P_sim(t:t+N-1);
    else
        P_hor = [P_sim(t:end); zeros(t+N-1-Tsim,1)];
    end
    
    % Solve MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; P_hor] ...
    );
    
    w_opt = full(sol.x);
    
    % Optimal control extraction
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);
    
    % Apply real dynamics
    xk = full(f(xk,uk,P_sim(t)));
    
    % Store data
    theta_hist(t) = xk;
    u_hist(t) = uk;
    p_hist(t) = P_sim(t);
    
    % Receding Horizon
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Plotting Results
% --------------------------
time = 1:Tsim;
fs = 18;          % Master font size for readability
fs_tit = fs + 4;  % Title font size
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];
stdRed   = [0.85, 0.325, 0.098]; % Rain color

% Create large figure
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;

% Reference
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Moisture Constraints
yline(0.15, 'k:', 'Minimum (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Maximum (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture Level', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: MPC Control Action ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;

% Irrigation Constraint
yline(6.0, 'r:', 'Irrigation Limit (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: External Disturbance (Rain) ---
subplot(3,1,3)
stairs(time, p_hist, 'Color', stdRed, 'LineWidth', 3);
ylim([-0.5, max(p_hist) + 4]); 
ylabel('Rain (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('External Disturbance (Rain)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
