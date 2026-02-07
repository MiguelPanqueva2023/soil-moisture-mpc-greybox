addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clc; clear; close all
import casadi.*

% At this stage, the Model Predictive Controller (MPC) was integrated with the 
% physical water balance model to automatically generate a synthetic dataset 
% under realistic operating conditions. 
% To achieve this, the MPC calculates the optimal irrigation action at each 
% time step using the hydrological model, while the "real plant" is simulated 
% as a perturbed version of the same model, incorporating unmodeled dynamics 
% and noise. The difference between the simulated real plant response and the 
% physical model defines a residual term, which is stored along with the 
% state variables, control actions, and climatic disturbances. 
% This procedure allows for the construction of a consistent dataset to 
% subsequently train a residual neural network, laying the foundation for a 
% Grey-Box model that combines physical knowledge and machine learning.

%% ===============================
% PHYSICAL PARAMETERS
%% ===============================
Zr = 0.30;       % Root depth [m]
dt = 1;          % Time step [day]
theta_wp = 0.12; % Wilting point
theta_fc = 0.25; % Field capacity
theta_ref = 0.25;% Desired reference

%% ===============================
% MPC SETTINGS
%% ===============================
N = 10;          % Horizon
Tsim = 200;      % Total simulation days

%% ===============================
% CASADI MODEL DEFINITION
%% ===============================
x = SX.sym('x');     % State: soil moisture
u = SX.sym('u');     % Control: irrigation
p = SX.sym('p',2);   % Parameters: [P; ETc]
P_rain = p(1);
ETc    = p(2);

% Water Stress factor
Ks = fmin(1, fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

% Drainage
D = fmax(0, x-theta_fc)*5;

% Dynamics (Euler integration)
x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% ===============================
% MPC OPTIMIZATION VARIABLES
%% ===============================
X = SX.sym('X',N+1);
U = SX.sym('U',N);
X0 = SX.sym('X0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

J = 0; g = [];
g = [g; X(1)-X0]; % Initial state constraint

for k=1:N
    e = X(k)-theta_ref;
    J = J + 10*e^2 + 0.01*U(k)^2; % Objective function
    g = [g; X(k+1)-f(X(k),U(k),[P_h(k);ETc_h(k)])]; % Model constraints
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h]);
solver = nlpsol('solver','ipopt',OPT);

%% ===============================
% BOUNDS (CONSTRAINTS)
%% ===============================
lbg = zeros(size(g));
ubg = zeros(size(g));
lbw = [0.1*ones(N+1,1); zeros(N,1)]; % Min moisture and irrigation
ubw = [0.4*ones(N+1,1); 6*ones(N,1)];  % Max moisture and irrigation

%% ===============================
% SIMULATION AND DATASET GENERATION
%% ===============================
theta_k = 0.18;
w0 = zeros((N+1)+N,1);

% Dataset storage
Xdata = zeros(Tsim,4); % Inputs: [theta, u, P, ETc]
Ydata = zeros(Tsim,1); % Target: Residual (theta_real - theta_model)

theta_model_hist = zeros(Tsim,1);
theta_real_hist  = zeros(Tsim,1);
u_hist = zeros(Tsim,1);

for t=1:Tsim
    % Climatic disturbances
    ETc_t = 4 + 0.5*sin(2*pi*t/40);
    P_t   = (t>40 & t<60)*4 + (t>120 & t<140)*3;
    
    P_hor  = P_t*ones(N,1);
    ETc_hor= ETc_t*ones(N,1);
    
    % Solve MPC for optimal irrigation
    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[theta_k;P_hor;ETc_hor]);
    
    w_opt = full(sol.x);
    Uopt = w_opt(N+2:end);
    u_k = Uopt(1);
    
    % NOMINAL PHYSICAL MODEL
    theta_model = full(f(theta_k,u_k,[P_t;ETc_t]));
    
    % REAL PLANT (Simulated with unknown dynamics/noise)
    % This represents the "unmodeled" part of the system
    residual = 0.02*sin(5*theta_k) + 0.01*theta_k.^2 + 0.005*randn;
    theta_real = theta_model + residual;
    
    % SAVE TO DATASET
    Xdata(t,:) = [theta_k u_k P_t ETc_t];
    Ydata(t) = theta_real - theta_model; % The residual we want the NN to learn
    
    % Close the loop using the REAL moisture
    theta_k = theta_real;
    
    % Logging for plots
    theta_model_hist(t)=theta_model;
    theta_real_hist(t)=theta_real;
    u_hist(t)=u_k;
    w0 = w_opt;
end

%% ===============================
% PLOTTING
%% ===============================
t_vec = 1:Tsim;
fs = 18;          % Master font size
fs_tit = fs + 4;  
black = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Moisture (Model vs Real) ---
subplot(3,1,1)
plot(t_vec, theta_model_hist, 'b', 'LineWidth', 3); hold on;
plot(t_vec, theta_real_hist, 'r--', 'LineWidth', 2.5);

% Limit labels
yline(theta_wp, 'k:', 'Wilting Point', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
yline(0.40, 'k:', 'Saturation', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

ylim([0.05, 0.55]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture (Dataset Generation)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

lgdd = legend('Model', 'Real Plant', 'Location', 'northeast');
lgdd.FontSize = fs-2;
lgdd.TextColor = black;
lgdd.Color = 'w';
lgdd.EdgeColor = black;

% --- Subplot 2: MPC Control Action (Irrigation) ---
subplot(3,1,2)
stairs(t_vec, u_hist, 'k', 'LineWidth', 3); hold on;

% Irrigation limit
yline(6.0, 'r:', 'Max Irrigation (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

% --- Subplot 3: Real Residual (NN Target) ---
subplot(3,1,3)
plot(t_vec, Ydata, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 3); % Green color for residual
ylim([min(Ydata)-0.05, max(Ydata)+0.05]); 
ylabel('\Delta\theta (Residual)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Real Residual (Neural Network Target)', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

%% ===============================
% DATASET EXPORT
%% ===============================
save greybox_dataset.mat Xdata Ydata
disp('Dataset saved: greybox_dataset.mat')