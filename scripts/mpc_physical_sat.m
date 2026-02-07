%% MPC for Soil Water Balance
% Conceptual model with soft saturations
% Objective: Structural validation of the MPC (without grey-box yet)
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% An integral state of the moisture error was introduced to eliminate the 
% steady-state error observed in the original MPC. This ensures exact 
% reference tracking, even in the presence of constant disturbances.

% Initially, the MPC presented a stationary error due to constant losses 
% from evapotranspiration and drainage. To solve this, integral action was 
% incorporated into the MPC formulation, allowing the elimination of permanent 
% error and achieving exact tracking of the moisture reference, even under 
% climatic disturbances.

% Soft physical saturations were introduced to maintain hydrological 
% consistency without compromising the feasibility of the problem.
% Physical saturations for moisture and irrigation were added, with limits 
% or penalties for being out of range to improve stability and physical 
% interpretability. Theta is smoothed with a softplus function in CasADi 
% to eliminate discontinuities; additionally, it prevents aggressive changes 
% in irrigation and better represents real irrigation systems.

%% -------------------------
% Soil Physical Parameters
% --------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]
theta_wp  = 0.12; % Wilting point
theta_fc  = 0.25; % Field capacity
theta_sat = 0.35; % Saturation
theta_ref = 0.25; % Desired reference

%% -------------------------
% MPC Horizon
% --------------------------
N = 10;

%% -------------------------
% Cost Weights
% --------------------------
w_theta = 10;     % Tracking weight
w_u     = 0.01;   % Irrigation energy weight
w_du    = 0.5;    % Irrigation smoothness weight
w_int   = 5;      % Integral action weight
w_low   = 50;     % Lower saturation penalty
w_high  = 50;     % Upper saturation penalty
alpha_sat = 50;   % Smoothness of saturations (alpha)

%% -------------------------
% Smooth Functions
% --------------------------
softplus = @(x) (1/alpha_sat)*log(1 + exp(alpha_sat*x));

%% -------------------------
% Symbolic Model
% --------------------------
x  = SX.sym('x');    % Moisture (State)
u  = SX.sym('u');    % Irrigation (Control)
p  = SX.sym('p',2);  % [Precipitation; ETc]
P_rain = p(1);
ETc    = p(2);

% ETc dependent on theta (Simple water stress)
Ks = fmin(1, fmax(0, (x - theta_wp)/(theta_fc - theta_wp)));
ETc_eff = Ks * ETc;

% Drainage dependent on theta (smoothed)
K_d = 10;
D_theta = softplus(x - theta_fc) * K_d;

% Dynamic Model (Euler)
x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D_theta);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% Optimization Variables
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);
I = SX.sym('I',N+1); % Integral state
X0 = SX.sym('X0');
I0 = SX.sym('I0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

%% -------------------------
% Objective Function and Constraints
% --------------------------
J = 0;
g = [];

% Initial conditions
g = [g;
     X(1) - X0;
     I(1) - I0];

for k = 1:N
    % Error
    e = X(k) - theta_ref;
    
    % Main Cost Function
    J = J + w_theta*e^2 + w_u*U(k)^2 + w_int*I(k)^2;
    
    % Soft Moisture Saturations
    J = J + w_low  * softplus(theta_wp - X(k))^2;
    J = J + w_high * softplus(X(k) - theta_fc)^2;
    
    % Irrigation Smoothness (Rate of change penalty)
    if k > 1
        J = J + w_du*(U(k) - U(k-1))^2;
    end
    
    % System Dynamics (State and Integral)
    g = [g;
         X(k+1) - f(X(k), U(k), [P_h(k); ETc_h(k)]);
         I(k+1) - (I(k) + e)];
end

%% -------------------------
% NLP Formulation
% --------------------------
OPT_variables = [X; U; I];
OPT = struct('x',OPT_variables, ...
             'f',J, ...
             'g',g, ...
             'p',[X0; I0; P_h; ETc_h]);

opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Bounds (Constraints)
% --------------------------
lbg = zeros(size(g));
ubg = zeros(size(g));

lbw = [ ...
    0.05*ones(N+1,1);      % Theta min
    0.0*ones(N,1);         % Irrigation min
    -inf*ones(N+1,1) ];    % Integral state min
ubw = [ ...
    0.40*ones(N+1,1);      % Theta max
    6.0*ones(N,1);         % Irrigation max
    inf*ones(N+1,1) ];     % Integral state max

%% -------------------------
% Closed-loop Simulation
% -------------------------
Tsim = 60;
xk = 0.18;
ik = 0;
theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
ETc_hist   = zeros(Tsim,1);
P_hist     = zeros(Tsim,1);

% Warm-start initialization
w0 = zeros(length(OPT_variables),1);

for t = 1:Tsim
    % Disturbances profile
    ETc_t = 4 + sin(2*pi*t/30);
    P_t   = (t>=10 && t<=12)*4 + (t>=20 && t<=22)*4;
    
    % Solve NLP
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; ik; P_t*ones(N,1); ETc_t*ones(N,1)] );
    
    w_opt = full(sol.x);
    U_opt = w_opt(N+2:N+1+N);
    uk = U_opt(1);
    
    % Update System (Plant)
    xk = full(f(xk, uk, [P_t; ETc_t]));
    ik = ik + (xk - theta_ref);
    
    % Store data
    theta_hist(t) = xk;
    u_hist(t)     = uk;
    ETc_hist(t)   = ETc_t;
    P_hist(t)     = P_t;
    
    % Receding horizon
    w0 = w_opt;
end

%% -------------------------
% Plotting Results
% --------------------------
time = 1:Tsim;
fs = 18;          % Master font size
fs_tit = fs + 4;  % Title size
black = [0, 0, 0];
midBlue = [0, 0.4, 0.8];
stdBlue = [0, 0.447, 0.741];
stdRed  = [0.85, 0.325, 0.098];

% Create wide figure
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture (State) ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', midBlue, 'LineWidth', 3); hold on;

% Reference Line (Exact tracking due to integral action)
yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Moisture Constraints (0.05 - 0.40)
yline(0.05, 'k:', 'Min Opt. (0.05)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.40, 'k:', 'Max Opt. (0.40)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0, 0.45]); 
ylabel('$\theta$', 'Interpreter', 'latex', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture (MPC with Integral Action)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: MPC Control Action (Irrigation) ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', black, 'LineWidth', 3); hold on;

% Irrigation Limit Constraint
yline(6.0, 'r:', 'Irrigation Limit (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 7.5]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action (Smoothed)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Climatic Disturbances ---
subplot(3,1,3)
plot(time, ETc_hist, 'Color', stdBlue, 'LineWidth', 3); hold on;
stairs(time, P_hist, 'Color', stdRed, 'LineWidth', 3);

% Legend
lgd = legend('ET(t)', 'Precipitation');
lgd.FontSize = fs;
lgd.TextColor = black;
lgd.Color = 'w';
lgd.EdgeColor = black;

ylim([-0.5, max(ETc_hist) + 4]); 
ylabel('mm/day', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Climatic Disturbances', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
