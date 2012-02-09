%% Initialization
clear; close all; clc

% correct,outcome,user_id,question_id,question_type,group_name,track_name,subtrack_name,tag_string,round_started_at,answered_at,deactivated_at,answer_id,game_type,num_players,date_of_test,question_set_id
% 0,2,85818,5560,0,1,5,14,222 233 240 246,2010-08-18 20:17:13,2010-08-18 20:18:18,2010-08-18 20:18:18,6540,7,1,NULL,1567
% 1,1,85818,4681,0,1,5,0,24 49,2010-08-18 20:19:12,2010-08-18 20:20:34,2010-08-18 20:20:34,4742,7,1,NULL,1227
% 1,1,85818,1529,0,1,5,0,31 49,2010-08-18 20:20:42,2010-08-18 20:21:56,2010-08-18 20:21:56,4309,7,1,NULL,1148

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Read raw data
data = csvread('../../data/sample_20k.csv');

% Remove top row
data = data(2:end, :);

m = size(data, 1);
y = data(:, 1);
X = [data(:, 1) data(:,3:8) data(:,13:15) data(:,17)]; % TEST ONLY - See if the algorithm figures out that the answer is in the set
% X = [data(:,3:8) data(:,13:15) data(:,17)];
m = size(X, 1)

% Randomize data, set 80% to train and 20% to test
rand_indices = randperm(m);
train_indices = rand_indices(1:floor(m*0.8));
test_indices = rand_indices((floor(m*0.8)+1):end);
Xtrain = X(train_indices, :);
ytrain = y(train_indices, :);
Xtest = X(test_indices, :);
ytest = y(test_indices, :);

fprintf('\nTraining Logistic Regression...\n')

lambda = 1.0;
[theta] = train(Xtrain, ytrain, lambda, 250)

pred = sigmoid([ones(size(Xtest,1), 1) Xtest] * theta) > 0.5;

fprintf('Sample of actual vs. predicted:\n');
[ytest pred](1:10,:)

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

save -binary theta.mat theta

