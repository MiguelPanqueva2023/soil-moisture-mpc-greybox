%% MPC for Soil Water Balance
% Conceptual model with soft saturations
% Objective: Structural validation of the MPC (without grey-box yet)
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

% Precipitation was not modeled as an ideal step function, but as a double 
% sigmoid function representing the leading edge, core, and dissipation of 
% a storm. This allows for a physically realistic and differentiable climatic 
% excitation, suitable for predictive control. 

% The precipitation was modeled using a double sigmoid function that represents 
% the formation, development, and dissipation of a rain event. This avoids 
% artificial discontinuities and allows the MPC to exploit its predictive 
% capability by anticipating the arrival of water into the soil–plant system.

%% -------------------------
% Soil Physical Parameters
% --------------------------
Zr = 0.30;        % Root depth [m]
dt = 1.0;         % Time step [day]
theta_wp  = 0.12; % Wilting point
theta_fc  = 0.25; % Field capacity
theta_sat = 0.35; % Saturation
theta_ref = 0.25; % Target reference

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
w_low   = 50;     % Lower saturation weight
w_high  = 50;     % Upper saturation weight
alpha_sat = 50;   % Saturation smoothness (alpha)

%% -------------------------
% Smooth Functions
% --------------------------
softplus = @(x) (1/alpha_sat)*log(1 + exp(alpha_sat*x));

%% -------------------------
% Symbolic Model
% --------------------------
x  = SX.sym('x');    % Moisture
u  = SX.sym('u');    % Irrigation
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
    
    % Main Cost
    J = J + w_theta*e^2 + w_u*U(k)^2 + w_int*I(k)^2;
    
    % Soft moisture saturations
    J = J + w_low  * softplus(theta_wp - X(k))^2;
    J = J + w_high * softplus(X(k) - theta_fc)^2;
    
    % Irrigation smoothness
    if k > 1
        J = J + w_du*(U(k) - U(k-1))^2;
    end
    
    % Dynamics
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
    0.05*ones(N+1,1);      % theta min
    0.0*ones(N,1);         % irrigation min
    -inf*ones(N+1,1) ];    % integral min
ubw = [ ...
    0.40*ones(N+1,1);      % theta max
    6.0*ones(N,1);         % irrigation max
    inf*ones(N+1,1) ];     % integral max

%% -------------------------
% Closed-loop Simulation
% --------------------------
Tsim = 60;
xk = 0.18;
ik = 0;
theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
ETc_hist   = zeros(Tsim,1);
P_hist     = zeros(Tsim,1);
w0 = zeros(length(OPT_variables),1);

for t = 1:Tsim
    % Disturbances
    ETc_t = 4 + sin(2*pi*t/30);
    
    % Physical Rain (Double Sigmoid)
    % -------------------------
    Pmax = 4;       % mm/day (peak intensity)
    t1   = 10;      % event start
    t2   = 13;      % event end
    tau  = 0.5;     % front smoothness (days)
    
    sig = @(x) 1./(1 + exp(-x));
    
    P_event1 = Pmax * ( sig((t - t1)/tau) - sig((t - t2)/tau) );
    
    % Second rain event
    t3 = 20;
    t4 = 23;
    P_event2 = Pmax * ( sig((t - t3)/tau) - sig((t - t4)/tau) );
    
    P_t = P_event1 + P_event2;
    
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
    
    % Update system
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
% Plotting
% --------------------------
time = 1:Tsim;
fs = 18;          % Large font size
fs_tit = fs + 4;  % Larger titles
black = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture ---
subplot(3,1,1)
plot(time, theta_hist, 'b', 'LineWidth', 3); hold on;

yline(theta_ref, 'r--', 'Reference', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Control Action ---
subplot(3,1,2)
stairs(time, u_hist, 'k', 'LineWidth', 3); hold on;

yline(6.0, 'r:', 'Limit (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Disturbances ---
subplot(3,1,3)
plot(time, ETc_hist, 'LineWidth', 3); hold on;
plot(time, P_hist, 'LineWidth', 3);

ylim([-0.5, max([max(ETc_hist), max(P_hist)]) + 3]); 
ylabel('mm/day', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Climatic Disturbances', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');

lgd = legend('ETc(t)', 'Precipitation');
lgd.FontSize = fs;
lgd.TextColor = black;
lgd.Color = 'w';
lgd.EdgeColor = black;

grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridAlpha', 0.2, 'LineWidth', 1.5);
