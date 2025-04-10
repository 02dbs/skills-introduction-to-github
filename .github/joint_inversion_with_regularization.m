% joint_inversion_with_regularization.m
% Implementation of joint inversion with NSS residuals, magnetic field residuals, and regularization.
% Incorporates data normalization, noise estimates, forward modeling, and optimization.

% Set random number seed for reproducibility
rng(0);

% Step 1: Data preparation and preprocessing ---------------------------------
% Define synthetic observation data
[d_NSS_obs, d_x_obs, d_y_obs, d_z_obs, noise_std_NSS, noise_std_B] = prepare_synthetic_data();

% Normalize data
d_NSS_obs_norm = d_NSS_obs / max(abs(d_NSS_obs(:)));
d_x_obs_norm = d_x_obs / max(abs(d_x_obs(:)));
d_y_obs_norm = d_y_obs / max(abs(d_y_obs(:)));
d_z_obs_norm = d_z_obs / max(abs(d_z_obs(:)));

% Step 2: Forward modeling --------------------------------------------------
% Define grid and geological model
[X, Y, Z] = meshgrid(linspace(-1, 1, 20), linspace(-1, 1, 20), linspace(-1, 1, 20)); % 20x20x20 grid
geobody = define_geobody();

% Forward model for magnetic field components and NSS
[G_x, G_y, G_z, G_NSS] = forward_model(geobody, X, Y, Z);

% Step 3: Joint inversion objective function --------------------------------
% Define weights
w1 = 1; % Weight for NSS residuals
w2 = 1; % Weight for magnetic field residuals
lambda = 1e-2; % Regularization parameter

% Define objective function
objective_function = @(m) joint_objective_function(m, d_NSS_obs_norm, d_x_obs_norm, d_y_obs_norm, d_z_obs_norm, ...
    G_x, G_y, G_z, G_NSS, noise_std_NSS, noise_std_B, w1, w2, lambda);

% Step 4: Optimization ------------------------------------------------------
% Initial guess (ensure it matches the grid size)
m_init = rand(size(X)); % Random initial model

% Optimization configuration
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxIterations', 100, 'SpecifyObjectiveGradient', false);
[m_optimized, fval] = fminunc(objective_function, m_init, options);

% Step 5: Visualization -----------------------------------------------------
% Display results
visualize_results(m_optimized, G_x, G_y, G_z, G_NSS, d_NSS_obs_norm, d_x_obs_norm, d_y_obs_norm, d_z_obs_norm);

% End of main script --------------------------------------------------------

% -------------------------------------------------------------------------
% Functions
% -------------------------------------------------------------------------

% Function to prepare synthetic data
function [d_NSS, d_x, d_y, d_z, noise_std_NSS, noise_std_B] = prepare_synthetic_data()
    % Generate synthetic magnetic field data (x, y, z components)
    d_x = randn(20, 20, 20); % Example synthetic data
    d_y = randn(20, 20, 20);
    d_z = randn(20, 20, 20);
    
    % Generate NSS data from magnetic field gradients
    [grad_x, grad_y, grad_z] = gradient(d_x); % Gradients of x-component
    NSS = sqrt(grad_x.^2 + grad_y.^2 + grad_z.^2); % Frobenius norm of gradient tensor

    % Add noise
    noise_std_B = 0.05; % Standard deviation of noise for magnetic field
    noise_std_NSS = 0.02; % Standard deviation of noise for NSS
    d_x = d_x + noise_std_B * randn(size(d_x)); % Add noise to x-component
    d_y = d_y + noise_std_B * randn(size(d_y)); % Add noise to y-component
    d_z = d_z + noise_std_B * randn(size(d_z)); % Add noise to z-component
    NSS = NSS + noise_std_NSS * randn(size(NSS)); % Add noise to NSS

    % Output
    d_NSS = NSS;
end

% Function to define a geological body
function geobody = define_geobody()
    % Example geological body with magnetic properties
    geobody.vertices = [0, 0, 0; 1, 0, 0; 1, 1, 0; 0, 1, 0; ...
                        0, 0, 1; 1, 0, 1; 1, 1, 1; 0, 1, 1];
    geobody.faces = [1, 2, 3, 4; 5, 6, 7, 8; 1, 5, 8, 4; ...
                     2, 6, 7, 3; 3, 7, 8, 4; 1, 5, 6, 2];
    geobody.magnetization = rand(1, 3); % Random magnetization vector
end

% Function to perform forward modeling
function [G_x, G_y, G_z, G_NSS] = forward_model(geobody, X, Y, Z)
    % Placeholder for forward model (example data, replace with real calculations)
    G_x = sin(2 * pi * X) .* cos(2 * pi * Y) .* Z;
    G_y = cos(2 * pi * X) .* sin(2 * pi * Y) .* Z;
    G_z = sin(2 * pi * X) .* sin(2 * pi * Y) .* Z;

    % Compute NSS from gradients
    dx = X(2) - X(1); % Grid spacing in X direction
    dy = Y(2) - Y(1); % Grid spacing in Y direction
    dz = Z(2) - Z(1); % Grid spacing in Z direction

    % Calculate gradients for the Frobenius norm
    [grad_x, ~, ~] = gradient(G_x, dx, dy, dz);
    [~, grad_y, ~] = gradient(G_y, dx, dy, dz);
    [~, ~, grad_z] = gradient(G_z, dx, dy, dz);

    % Frobenius norm of the gradient tensor
    G_NSS = sqrt(grad_x.^2 + grad_y.^2 + grad_z.^2);

    % Ensure output sizes match
    assert(isequal(size(G_x), size(G_y), size(G_z), size(G_NSS)), 'Forward model outputs must have the same size.');
end

% Function to compute joint objective function
function obj = joint_objective_function(m, d_NSS, d_x, d_y, d_z, G_x, G_y, G_z, G_NSS, noise_std_NSS, noise_std_B, w1, w2, lambda)
    % Ensure m is the same size as the model grid
    if ~isequal(size(m), size(G_x))
        error('Input model m must match the grid size of the forward model.');
    end

    % Compute residuals
    NSS_residuals = (d_NSS - G_NSS) ./ noise_std_NSS;
    x_residuals = (d_x - G_x) ./ noise_std_B;
    y_residuals = (d_y - G_y) ./ noise_std_B;
    z_residuals = (d_z - G_z) ./ noise_std_B;

    % Compute data misfit term
    data_misfit = w1 * sum(NSS_residuals(:).^2) + ...
                  w2 * (sum(x_residuals(:).^2) + sum(y_residuals(:).^2) + sum(z_residuals(:).^2));

    % Compute regularization term (Tikhonov regularization)
    L = del2(m); % Laplacian operator for smoothness
    regularization = lambda * sum(L(:).^2);

    % Total objective function
    obj = data_misfit + regularization;
end

% Function to visualize results
function visualize_results(m, G_x, G_y, G_z, G_NSS, d_NSS, d_x, d_y, d_z)
    figure;
    subplot(2, 3, 1); imagesc(squeeze(mean(d_NSS, 3))); title('Observed NSS');
    subplot(2, 3, 2); imagesc(squeeze(mean(G_NSS, 3))); title('Predicted NSS');
    subplot(2, 3, 3); imagesc(squeeze(mean(m, 3))); title('Inverted Model');
end