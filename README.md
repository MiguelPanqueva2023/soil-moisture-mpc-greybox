# soil-moisture-mpc-greybox
Model Predictive Control of soil moisture using a water balance model with climatic disturbances, nonlinear dynamics, and a Grey-Box residual neural network.

# 🌱 MPC-Based Soil Moisture Control with Grey-Box Modeling

This repository contains the full implementation of a **Model Predictive Control (MPC)** framework for soil moisture regulation based on a dynamic soil water balance model, progressively extended with climatic disturbances, nonlinear hydrological processes, physical constraints, and a **Grey-Box residual neural network**.

The objective is to study how irrigation can be optimally managed under realistic conditions by combining **physical modeling** and **data-driven correction**.

---

## 📌 Project Overview

The system regulates the soil volumetric water content $\theta(k)$ using irrigation as the control variable.  
The MPC minimizes deviations from a desired moisture reference while respecting physical and operational constraints.

The project evolves through several modeling stages:

- **Base physical model**  
  Soil moisture dynamics with constant evapotranspiration and drainage.

- **Precipitation as external disturbance**  
  Rainfall enters the water balance as an exogenous input.

- **Time-varying evapotranspiration**  
  Climatic demand varies with time.

- **Soil-dependent evapotranspiration**  
  A water-stress coefficient $K_s(\theta)$ modulates crop water uptake.

- **Nonlinear deep drainage**  
  Drainage activates when soil moisture exceeds field capacity.

- **Smooth nonlinearities and physical saturations**  
  Hard discontinuities are replaced by smooth functions (softplus, sigmoids) and physical bounds are enforced.

- **Integral action**  
  Eliminates steady-state errors under persistent disturbances.

- **Sigmoidal rainfall profile**  
  Rainfall is modeled with a smooth double-sigmoid function instead of step functions.

- **Variable moisture reference and climatic scenarios**  
  The controller tracks time-varying crop water demand under different climate regimes.

- **Grey-Box residual model**  
  A neural network learns unmodeled dynamics and corrects the physical model.

---

## 🧠 Grey-Box Concept

The Grey-Box structure combines:

- A **physical model** (soil water balance)  
- A **neural network residual** that learns the mismatch between reality and the physical model  

The Grey-Box model is defined as:

**θ(k+1) = f_phys(θ(k), u(k), P(k), ET(k)) + r_NN(θ(k), u(k), P(k), ET(k))**

where:
- `f_phys(·)` is the physical soil water balance model
- `r_NN(·)` is the neural network residual


This allows the controller to retain physical interpretability while increasing accuracy.

---

## 📂 Repository Structure

```text
/scripts
├── mpc_base.m
├── mpc_rain.m
├── mpc_ET_time.m
├── mpc_ET_theta.m
├── mpc_drainage.m
├── mpc_smooth.m
├── mpc_physical_sat.m
├── mpc_sigmoid_rain_model.m
├── mpc_scenarios.m
├── generate_dataset.m
├── train_residual_nn.m
└── mpc_greybox.m

/data
└── greybox_dataset.mat

/models
└── NN_residual.mat

/figures
└── simulation_plots
```

---

## ⚙️ Requirements

- MATLAB (R2018b or later recommended)  
- CasADi (tested with v3.7.x)  
- IPOPT (via CasADi)

---

## ▶️ How to Run

### 1️⃣ Run the MPC with physical model

Start from the simplest model and move forward:

```matlab
mpc_base
mpc_rain
mpc_ET_time
mpc_ET_theta
mpc_drainage
mpc_smooth
mpc_physical_sat
mpc_sigmoid_rain_model
mpc_scenarios
```

2️⃣ Generate synthetic data
generate_dataset

This produces:
/data/greybox_dataset.mat

3️⃣ Train the residual neural network
train_residual_nn

This produces:
/models/NN_residual.mat

4️⃣ Run Grey-Box MPC
mpc_greybox

📊 What the Simulations Show

The MPC successfully regulates soil moisture under:

Rainfall

• Variable evapotranspiration

• Nonlinear drainage

• Changing crop demand

The Grey-Box version improves tracking and robustness by compensating for unmodeled dynamics.

📚 Scientific Motivation

This framework demonstrates how control theory and machine learning can be combined to improve irrigation management under uncertainty, making it suitable for:

• Precision agriculture

• Smart irrigation systems

• Climate-adaptive water management

