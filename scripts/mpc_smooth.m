%% MPC for Soil Water Balance
% Smoothed non-linear physical model
% ETc and drainage dependent on soil moisture
% Initial Implementation in CasADi + MATLAB
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% A nonlinear predictive control scheme based exclusively on a physical 
% model of the soil water balance, incorporating smoothed nonlinearities 
% in evapotranspiration and drainage.
% The controller converges to a stable equilibrium below the reference. 
% This behavior is consistent with the current optimization problem 
% formulation, where the cost associated with irrigation dominates over 
% the tracking error. The result confirms the stability of the MPC scheme 
% and highlights the need for weight and/or constraint adjustments to 
% force reference tracking.

%% -------------------------
% Physical Parameters
% -------------------------
Zr = 0.30;          % Root depth [m]
dt = 1.0;           % Time step [day]
theta_ref = 0.25;   % Target moisture

%% -------------------------
% ETc Parameters (Smoothed)
% -------------------------
ETc_pot   = 4.0;    % Potential ETc [mm/day]
theta_crit = 0.20;  % Critical moisture
gamma     = 40;     % ETc sigmoid slope

%% -------------------------
% Drainage Parameters (Smoothed)
% -------------------------
theta_fc = 0.28;    % Field capacity
k_d      = 5.0;     % Drainage coefficient
alpha    = 40;      % Drainage sigmoid slope

%% -------------------------
% MPC Horizon
% -------------------------
N = 10;

%% -------------------------
% Symbolic Definition
% -------------------------
x = SX.sym('x');    % State: soil moisture
u = SX.sym('u');    % Control: irrigation

% Water stress factor (Simplified FAO-56)
Ks = 1/(1 + exp(-gamma*(x - theta_crit)));

% Theta-dependent Evapotranspiration
ETc = ETc_pot * Ks;

% Theta-dependent Drainage
D = k_d*(x - theta_fc)/(1 + exp(-alpha*(x - theta_fc)));

% Dynamic Model (Euler)
x_next = x + dt*(1/(Zr*1000))*(u - ETc - D);
f = Function('f',{x,u},{x_next});

%% -------------------------
% Optimization Variables
% -------------------------
X = SX.sym('X',N+1);   % States
U = SX.sym('U',N);     % Controls
X0 = SX.sym('X0');     % Initial state (parameter)

%% -------------------------
% Objective Function and Constraints
% -------------------------
J = 0;
g = [];

% Initial condition
g = [g; X(1) - X0];

for k = 1:N
    % Cost: tracking + irrigation penalty
    J = J + 10*(X(k)-theta_ref)^2 + 0.001*U(k)^2;
    % Dynamics
    g = [g; X(k+1) - f(X(k),U(k))];
end

%% -------------------------
% NLP Formulation
% -------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',X0);
opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Physical Limits
% -------------------------
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
% -------------------------
Tsim = 30;      % Days
xk = 0.18;      % Initial moisture
theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);

% Initialization
w0 = zeros((N+1)+N,1);

for t = 1:Tsim
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', xk ...
    );
    
    w_opt = full(sol.x);
    
    % Optimal control
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);
    
    % System evolution
    xk = full(f(xk,uk));
    
    % Save results
    theta_hist(t) = xk;
    u_hist(t) = uk;
    
    % Receding horizon
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Plots
% -------------------------
time = 1:Tsim;
fs = 18;          % Master font size (Large)
fs_tit = fs + 4;  % Title size
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];

% Create wide figure
figure('Color','w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

% --- Subplot 1: Soil Moisture ---
subplot(2,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;

% Reference Line
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Moisture Constraints (0.15 - 0.35)
yline(0.15, 'k:', 'Min. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Max. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

% Adjustment for visibility of reference gap
ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture (Smoothed Non-linear Model)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: MPC Control Action ---
subplot(2,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;

% Maximum Irrigation Constraint
yline(6.0, 'r:', 'Irrigation Limit (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
