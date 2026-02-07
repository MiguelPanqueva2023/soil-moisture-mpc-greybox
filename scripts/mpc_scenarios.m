%% Hydrological MPC with Dynamic Reference and Realistic Climate
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

%% -------------------------
% Soil Parameters
% --------------------------
Zr = 0.30;           % Root depth [m]
dt = 1.0;            % Time step [day]
theta_wp  = 0.12;    % Wilting point
theta_fc  = 0.25;    % Field capacity
theta_sat = 0.40;    % Saturation

%% -------------------------
% MPC Horizon
% --------------------------
N = 10;

%% -------------------------
% Weights
% --------------------------
w_theta = 50;        % State tracking weight
w_u     = 0.01;      % Irrigation energy weight
w_du    = 0.3;       % Irrigation rate of change weight (smoothness)

alpha = 50;          % Softplus smoothness factor
softplus = @(x) (1/alpha)*log(1 + exp(alpha*x));

%% -------------------------
% Symbolic Variables
% --------------------------
x = SX.sym('x');          % State: soil moisture
u = SX.sym('u');          % Control: irrigation
p = SX.sym('p',2);        % Parameters: [Precipitation, ETc]
P_rain = p(1);
ETc    = p(2);

%% -------------------------
% Water Stress factor Ks
% --------------------------
Ks = fmin(1, fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

%% -------------------------
% Smooth Drainage
% --------------------------
K_d = 10;
D = K_d*softplus(x-theta_fc);

%% -------------------------
% Dynamics
% --------------------------
x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% MPC variables
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);
X0   = SX.sym('X0');       % Initial state
P_h  = SX.sym('P_h',N);    % Precipitation horizon
ETc_h= SX.sym('ETc_h',N);  % ETc horizon
th_ref = SX.sym('th_ref',N); % Dynamic reference horizon

%% -------------------------
% Cost Function
% --------------------------
J = 0;
g = [];
g = [g; X(1)-X0]; % Initial condition constraint

for k=1:N
    e = X(k)-th_ref(k);
    J = J + w_theta*e^2 + w_u*U(k)^2;
    if k>1
        J = J + w_du*(U(k)-U(k-1))^2;
    end
    g = [g; X(k+1)-f(X(k),U(k),[P_h(k);ETc_h(k)])];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h;th_ref]);
solver = nlpsol('solver','ipopt',OPT,struct('ipopt',struct('print_level',0)));

%% -------------------------
% Bounds (Constraints)
% --------------------------
lbg = zeros(size(g)); ubg = lbg;
lbw = [0.10*ones(N+1,1); 0*ones(N,1)]; % Min moisture and irrigation
ubw = [0.40*ones(N+1,1); 6*ones(N,1)];  % Max moisture and irrigation

%% -------------------------
% Realistic Climate Generation
% --------------------------
Tsim = 60;
xk = 0.18;
ETc_hist = zeros(Tsim,1);
P_hist   = zeros(Tsim,1);
theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);
theta_ref_hist = zeros(Tsim,1);

t = (1:Tsim)';
ETc_hist = 4 + 1.2*sin(2*pi*t/30); % Oscillating ETc demand

% Double sigmoid for rain events
sig = @(t,t0,w) 1./(1+exp(-10*(t-t0))) - 1./(1+exp(-10*(t-(t0+w))));
P_hist = 3*sig(t,10,4) + 3*sig(t,20,4);

%% -------------------------
% Closed-loop Simulation
% --------------------------
w0 = zeros((N+1)+N,1);
for k=1:Tsim
    
    ETc_k = ETc_hist(k);
    P_k   = P_hist(k);
    
    % Dynamic Physiological Reference logic
    % Normalized ET (ETn) scales the target moisture between WP and FC
    ETn = min(1,max(0,(ETc_k-2)/(6-2)));
    th_ref_k = theta_wp + (theta_fc-theta_wp)*ETn;
    
    P_hor = P_k*ones(N,1);
    ETc_hor = ETc_k*ones(N,1);
    th_ref_hor = th_ref_k*ones(N,1);
    
    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[xk;P_hor;ETc_hor;th_ref_hor]);
    
    w_opt = full(sol.x);
    Uopt = w_opt(N+2:end);
    uk = Uopt(1);
    
    % Apply dynamics to the plant
    xk = full(f(xk,uk,[P_k;ETc_k]));
    
    theta_hist(k)=xk;
    u_hist(k)=uk;
    theta_ref_hist(k)=th_ref_k;
    
    w0=w_opt; % Warm start for next step
end

%% -------------------------
% Plotting Results
% --------------------------
time = 1:Tsim;
fs = 18;          % Master font size
fs_tit = fs + 4;  
black = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Soil Moisture (State) ---
subplot(3,1,1)
plot(t, theta_hist, 'b', 'LineWidth', 3); hold on;
plot(t, theta_ref_hist, 'r--', 'LineWidth', 2.5);

% Constraints labels
yline(theta_wp, 'k:', 'Wilting Point', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
yline(theta_sat, 'k:', 'Saturation', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

ylim([0.05, 0.50]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', black, 'FontWeight', 'bold');
title('Soil Moisture', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgdd = legend('\theta','\theta_{ref}(t)', 'Location', 'northeast');
lgdd.FontSize = fs-2;
lgdd.TextColor = black;
lgdd.Color = 'w';
lgdd.EdgeColor = black;

% --- Subplot 2: MPC Control Action (Irrigation) ---
subplot(3,1,2)
stairs(t, u_hist, 'k', 'LineWidth', 3); hold on;
yline(6, 'r:', 'Max Irrigation (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Irrigation (mm/day)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('MPC Control Action', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

% --- Subplot 3: Disturbances ---
subplot(3,1,3)
plot(t, ETc_hist, 'LineWidth', 3); hold on;
stairs(t, P_hist, 'LineWidth', 3);

ylim([-0.5, 11]); 
ylabel('mm/day', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
xlabel('Time (days)', 'FontSize', fs, 'Color', black, 'FontWeight', 'bold');
title('Climatic Disturbances', 'FontSize', fs_tit, 'Color', black, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', black, 'YColor', black, 'GridColor', black, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgd = legend('ETc(t)', 'Precipitation', 'Location', 'northeast');
lgd.FontSize = fs-2;
lgd.TextColor = black;
lgd.Color = 'w';
lgd.EdgeColor = black;
