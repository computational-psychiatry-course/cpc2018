function [posterior, out, result] = practical_VBA()

% design
% -------------------------------------------------------------------------
% Here we describe the stimuli of a simple delay discounting task in which
% participants have to choose between an low reward but immediate option (1
% euro today) and a higher reard but delayed option (eg. 4 euros in 15
% days)

% number of trials
N = 500;

% random trial conditions
low_reward = 1;
max_high_reward = 5;
max_delay = 30;

value_now = low_reward * ones (1, N); % value of the immediate option
value_delay = randi (max_high_reward, 1, N); % value of the delayed option
delay = randi (max_delay, 1, N); % actual delay in days

% model inputs (each column is a new trial)
u = [ value_now; 
      value_delay; 
      delay]; 


% model definition
% -------------------------------------------------------------------------
% Here we define our different hypotheses about how delay discounts value.
% We implement two competing models: hyperbolic and exponential
% discounting.

% observation function (hyperbolic)
    function g = g_discount_hyp (~, phi, u, ~)
        SV_delay = u(2) ./ (1 + phi * u(3));
        SV_now = u(1);   
        g = VBA_sigmoid (SV_delay - SV_now);
    end

% observation function (exponential)
    function g = g_discount_exp (~, phi, u, ~)
        SV_delay = u(2) * exp (- phi * u(3));
        SV_now = u(1);   
        g = VBA_sigmoid (SV_delay - SV_now);
    end

% simulation
% -------------------------------------------------------------------------
% In this section we simulate artificial data according to the hyperbolic
% model.

% parameters (delay discounting rate)
phi = 0.1;

% observation distribution
options = struct ();
options.sources.type = 1; % 0: gaussian, 1: binary, 2: categorical

% uncomment to split the data into two independent sessions
% options.multisession.split = [N/2 N/2]; 

% display options
options.verbose = false; % display text in the command window
options.DisplayWin = true; % display figures

% simulate data using hyperbolic discounting
fprintf('Simulating data using hyperbolic discounting with k = %3.2f\n',phi); 
y = VBA_simulate (N, [], @g_discount_hyp ,[], phi, u, [], [], options);

% inversion
% -------------------------------------------------------------------------
% In this section we estimate the parameters (posterior distribution) and
% the evidence for the two competing models

% model dimensions
dim.n_phi = 1;

% invert hyperbolic and exponential discounting model
[posterior(1), out(1)] = VBA_NLStateSpaceModel (y, u, [], @g_discount_hyp, dim, options);
[posterior(2), out(2)] = VBA_NLStateSpaceModel (y, u, [], @g_discount_exp, dim, options);

% parameter estimation error
estimation_error = posterior(1).muPhi - phi

% model selection
% -------------------------------------------------------------------------
% perform model selection to compare hyperbolic and exponential dicounting
% hypotheses. Of course, you would need to simulate more subjects and try
% different discount factors to really assess the validity of the design

% model x subject matrix of (approximate) model evidences
F = [out.F]'; 

% RAndom effect model selection
[p, o] = VBA_groupBMC(F, options);

% display statistics
[~, idxWinner] = max(o.Ef);
fprintf('The best model is the model %d: Ef = %4.3f (pxp = %4.3f)\n',idxWinner, o.Ef(idxWinner), o.pxp(idxWinner));

% display
% -------------------------------------------------------------------------
% It is ALWAYS a good idea to (1) plot your data and (2) plot your model
% predictions in a similar fashion. This is the best way to check how your
% different models make differential predictions about specific data
% patterns, and to which degree your data indeed support the best model.

% loop over conditions
for val = unique(value_delay)
    for d = unique(delay)
        % find corresponding trials
        trial_idx = find(u(2, :) == val & u(3, :) == d);
        if ~ isempty (trial_idx)
            % average response rate
            result.pr(val, d) = mean (y(trial_idx));
            % prediction (no need for average!)
            result.gx1(val, d) = out(1).suffStat.gx(trial_idx(1));
            result.gx2(val, d) = out(2).suffStat.gx(trial_idx(1));
        else
            result.pr(val, d) = nan;
            result.gx1(val, d) = nan;
            result.gx2(val, d) = nan;
        end
    end
end
       
% overlay data and model predictions
VBA_figure();

subplot (1, 2, 1); 
title ('hyperbolic model');
hold on;
plot (result.pr', 'o');
set (gca, 'ColorOrderIndex', 1);
plot (result.gx1');

subplot (1, 2, 2); 
title ('exponential model');
hold on;
plot (result.pr', 'o');
set (gca, 'ColorOrderIndex', 1);
plot (result.gx2');

end
