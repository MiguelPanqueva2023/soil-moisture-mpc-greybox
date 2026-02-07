clc; clear; close all
import casadi.*

%% ============================================================
% 1. LOAD TRAINED NETWORK
% ============================================================
load NN_residual.mat   % Load the weights and normalization parameters
% Contains: W1, b1, W2, b2, W3, b3, xmin, xmax, ymin, ymax

%% ============================================================
% 2. PHYSICAL MODEL
% ============================================================
Zr = 0.30;           % Root depth [m]
dt = 1;              % Time step [day]
theta_wp = 0.12;     % Wilting point
theta_fc = 0.25;     % Field capacity

x = SX.sym('x');     % Moisture state
u = SX.sym('u');     % Irrigation control
p = SX.sym('p',2);   % Parameters: [Rain, ETc]
P = p(1);
ETc = p(2);

% Water stress and effective ETc
Ks = fmin(1,fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

% Drainage model
K_d = 10;
D = log(1+exp(10*(x-theta_fc)));

% Dynamic model (Euler)
x_next = x + dt*(1/(Zr*1000))*(u + P - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% ============================================================
% 3. MPC SETUP
% ============================================================
N = 10;              % Prediction horizon
theta_ref = 0.25;    % Moisture reference

X = SX.sym('X',N+1);
U = SX.sym('U',N);
X0 = SX.sym('X0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

J = 0;
g = [X(1)-X0]; % Initial condition constraint

for k=1:N
    % Cost function: tracking error + control effort
    J = J + 10*(X(k)-theta_ref)^2 + 0.01*U(k)^2;
    % Dynamics constraints
    g = [g;
         X(k+1) - f(X(k),U(k),[P_h(k);ETc_h(k)])];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h]);
solver = nlpsol('solver','ipopt',OPT);

% Bounds
lbg = zeros(size(g)); ubg=lbg;
lbw = [0.1*ones(N+1,1); 0*ones(N,1)]; % Min theta and irrigation
ubw = [0.4*ones(N+1,1); 6*ones(N,1)];   % Max theta and irrigation

%% ============================================================
% 4. GREY-BOX SIMULATION
% ============================================================
T = 40;              % Simulation days
xk = 0.18;           % Initial moisture
theta_hist = zeros(T,1);
theta_phys = zeros(T,1);
u_hist = zeros(T,1);
w0 = zeros(N+1+N,1); % Warm start initialization

for t=1:T
    % Weather disturbances
    P_t = (t>=10 && t<=14)*4;
    ETc_t = 4 + sin(2*pi*t/30);
    P_hor = P_t*ones(N,1);
    ETc_hor = ETc_t*ones(N,1);
    
    % Solve NLP
    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[xk;P_hor;ETc_hor]);
    w_opt = full(sol.x);
    uk = w_opt(N+2); % Optimal irrigation for the first step
    
    % ---- PHYSICAL MODEL PREDICTION
    theta_f = full(f(xk,uk,[P_t;ETc_t]));
    
    % ---- RESIDUAL NEURAL NETWORK INFERENCE
    x_nn = [xk; uk; P_t; ETc_t];
    % Normalize input
    x_n = (x_nn - xmin)./(xmax - xmin + eps);
    % Forward pass
    a1 = tanh(W1*x_n + b1);
    a2 = tanh(W2*a1 + b2);
    r_n = W3*a2 + b3;
    % Denormalize output (residual)
    r = r_n*(ymax - ymin) + ymin;
    
    % ---- GREY-BOX INTEGRATION (Physics + ML)
    xk = theta_f + r;
    
    % Logging
    theta_hist(t) = xk;
    theta_phys(t) = theta_f;
    u_hist(t) = uk;
    w0 = w_opt;
end

%% ============================================================
% 5. PLOTTING
% ============================================================
t_vec = 1:T;
fs = 18;          % Master font size
fs_tit = fs + 4;  
black = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Model Comparison (Physics vs Grey-Box) ---
subplot(2,1,1)
plot(t_vec, theta_phys, '--b', 'LineWidth', 2.5); hold on; % Physical model (dashed blue)
plot(t_vec, theta_hist, 'r', 'LineWidth', 3);           % Grey-box (solid red)
yline(theta_ref, 'k--', 'LineWidth', 2);                % Reference (dashed black)

% Physical boundary labels
yline(theta_wp, 'k:', 'Wilting Point', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
yline(0.40, 'k:', 'Saturation', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

ylim([0.05, 0.50]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Effect of Residual Network (Grey-Box)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgd1 = legend('Physical Model', 'Grey-box', 'Reference', 'Location', 'northeast');
lgd1.FontSize = fs-2;
lgd1.TextColor = black;
lgd1.Color = 'w';
lgd1.EdgeColor = black;

% --- Subplot 2: Control Action ---
subplot(2,1,2)
stairs(t_vec, u_hist, 'k', 'LineWidth', 3); hold on;

% Max Irrigation limit
yline(6, 'r:', 'Max Irrigation (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);