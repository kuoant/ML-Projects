library(rethinking)
library(MASS)
library(ggplot2)


# Problem 1 ===================================================================#


# 1.a) ===========================#

data(eagles)
?eagles
print(eagles)

eagles$LargePirate <- ifelse(eagles$P=="L", 1, 0)
eagles$LargeVictim <- ifelse(eagles$V=="L", 1, 0)
eagles$AdultPirate <- ifelse(eagles$A=="A", 1, 0)

model_formula <- alist(
  y ~ dbinom(n, p),
  logit(p) <- a + bP * LargePirate + bV * LargeVictim + bA * AdultPirate,
  a  ~ dnorm(0, 1.5),
  bP ~ dnorm(0, 0.5),
  bV ~ dnorm(0, 0.5),
  bA ~ dnorm(0, 0.5)
)

model <- ulam(
  model_formula,
  data = eagles,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model)


# 1.b) ===========================#


# Extract posterior samples
post <- extract.samples(model)

# Predicted probabilities for each row
pred_probs <- sapply(1:nrow(eagles), function(i) {
  logit_p <- post$a + 
    post$bP * eagles$LargePirate[i] +
    post$bV * eagles$LargeVictim[i] +
    post$bA * eagles$AdultPirate[i]
  p <- logistic(logit_p)
  p
})

# Predicted success counts for each row
pred_counts <- sapply(1:nrow(eagles), function(i) {
  rbinom(n = nrow(post$a), size = eagles$n[i], prob = pred_probs[, i])
})

# Mean and 95% intervals for predicted probabilities
pred_prob_summary <- apply(pred_probs, 2, function(x) {
  c(mean = mean(x), HPDI(x, prob = 0.95))
})

# Mean and 95% intervals for predicted success counts
pred_count_summary <- apply(pred_counts, 2, function(x) {
  c(mean = mean(x), HPDI(x, prob = 0.95))
})

# Combine results into a data frame
pred_summary <- data.frame(
  MeanProb = pred_prob_summary[1, ],
  LowerProb = pred_prob_summary[2, ],
  UpperProb = pred_prob_summary[3, ],
  MeanCount = pred_count_summary[1, ],
  LowerCount = pred_count_summary[2, ],
  UpperCount = pred_count_summary[3, ]
)
print(pred_summary)


# Row labels
row_labels <- apply(eagles, 1, function(row) {
  paste0("LP:", row["LargePirate"], 
         " LV:", row["LargeVictim"], 
         " AP:", row["AdultPirate"])
})

# Plot predicted probabilities
plot(1:nrow(eagles), pred_summary$MeanProb, ylim=c(0,1), pch=16,
     xaxt="n", xlab="", 
     ylab="Predicted Probability",
     main="Predicted Probability with 95% Interval")
axis(1, at=1:nrow(eagles), labels=row_labels, las=2, cex.axis=0.7)  
arrows(1:nrow(eagles), pred_summary$LowerProb, 1:nrow(eagles), pred_summary$UpperProb,
       angle=90, code=3, length=0.05)

# Plot predicted counts with observed success counts
plot(1:nrow(eagles), pred_summary$MeanCount, ylim=range(c(pred_summary[,4:6], eagles$y)), pch=16,
     xaxt="n", xlab="", 
     ylab="Predicted Success Count",
     main="Predicted Success Count with 95% Interval")
axis(1, at=1:nrow(eagles), labels=row_labels, las=2, cex.axis=0.7) 
arrows(1:nrow(eagles), pred_summary$LowerCount, 1:nrow(eagles), pred_summary$UpperCount,
       angle=90, code=3, length=0.05)

# Add observed success counts as red dots
points(1:nrow(eagles), pred_summary$MeanCount, col="black", pch=16, cex=1.2)
points(1:nrow(eagles), eagles$y, col="red", pch=19, cex=1.2)
legend("topright", legend=c("Predicted Mean", "Observed Count"), pch=c(16, 19), col=c("black", "red"))






# 1.c) ===========================#


# Model 2 for comparison
model_formula2 <- alist(
    y ~ dbinom(n,p),
    logit(p) <- a + bP*LargePirate + bV*LargeVictim + bA*AdultPirate + bPA*LargePirate*AdultPirate,
    a ~ dnorm(0,1.5),
    bP ~ dnorm(0,0.5),
    bV ~ dnorm(0,0.5),
    bA ~ dnorm(0,0.5),
    bPA ~ dnorm(0,0.5)
  )

model2 <- ulam(
  model_formula2,
  data = eagles,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model2)

compare(model, model2)





# Problem 2 ===================================================================#


# 2.a) ===========================#

data(salamanders)
print(salamanders)

model_formula <- alist(
  SALAMAN ~ dpois(lambda),
  log(lambda) <- a + bP * PCTCOVER,
  a ~ dnorm(0, 1.5),
  bP ~ dnorm(0, 0.5)
)

model <- ulam(
  model_formula,
  data = salamanders,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model)


post <- extract.samples(model)

lambda_pred <- exp(outer(post$a, rep(1, nrow(salamanders))) + 
                     outer(post$bP, salamanders$PCTCOVER))

lambda_summary <- apply(lambda_pred, 2, function(x) PI(x, prob = 0.95))

count_pred <- sapply(1:ncol(lambda_pred), function(i) rpois(nrow(post$a), lambda_pred[,i]))
count_summary <- apply(count_pred, 2, function(x) PI(x, prob = 0.95))

results <- data.frame(
  PCTCOVER = salamanders$PCTCOVER,
  Lambda_Mean = apply(lambda_pred, 2, mean),
  Lambda_Lower = lambda_summary[1, ],
  Lambda_Upper = lambda_summary[2, ],
  Count_Mean = apply(count_pred, 2, mean),
  Count_Lower = count_summary[1, ],
  Count_Upper = count_summary[2, ],
  Observed_Count = salamanders$SALAMAN
)

print(results)

plot(results$PCTCOVER, results$Lambda_Mean, pch = 19, 
     xlab = "Percent Cover", ylab = "Predicted Count",
     main = "Predicted Salamander Count with 95% Interval",
     ylim = c(0, 8))
arrows(results$PCTCOVER, results$Lambda_Lower, 
       results$PCTCOVER, results$Lambda_Upper, angle=90, code=3, length=0.05)
points(salamanders$PCTCOVER, salamanders$SALAMAN, pch = 1, col = "red") 


# 2.b) ===========================#

# Model 2 for comparison
model_formula2 <- alist(
  SALAMAN ~ dpois(lambda),
  log(lambda) <- a + bP * PCTCOVER + bF * FORESTAGE,
  a ~ dnorm(0, 1),
  bP ~ dnorm(0, 1),
  bF ~ dnorm(0, 1)
)

model2 <- ulam(
  model_formula2,
  data = salamanders,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model2)




# Problem 3 ===================================================================#

data(NWOGrants)
print(NWOGrants)

disciplines <- unique(NWOGrants$discipline)
female_quota <- numeric(length(disciplines))
success_rate <- numeric(length(disciplines))

# Loop through each discipline and compute the values
for (i in 1:length(disciplines)) {
  # Filter data for the current discipline
  data <- NWOGrants[NWOGrants$discipline == disciplines[i], ]
  
  # Calculate total male and female applicants
  total_female_applicants <- sum(data$applications[data$gender == "f"])
  total_male_applicants <- sum(data$applications[data$gender == "m"])
  
  # Total applicants in the discipline
  total_applicants <- total_female_applicants + total_male_applicants
  
  # Calculate the female quota
  female_quota[i] <- total_female_applicants / total_applicants
  
  # Calculate the success rate
  total_awards <- sum(data$awards)
  success_rate[i] <- total_awards / total_applicants
}

# Create a data frame for the plot
plot_data <- data.frame(
  discipline = disciplines,
  female_quota = female_quota,
  success_rate = success_rate
)

# Scatter plot of female quota vs. success rate
ggplot(plot_data, aes(x = female_quota, y = success_rate)) +
  geom_point(color = "blue", size = 4) +
  labs(
    title = "Female Quota vs. Success Rate by Discipline",
    x = "Female Quota",
    y = "Success Rate"
  ) +
  theme_minimal()

# Success rate calculation
NWOGrants$success_rate <- NWOGrants$awards / NWOGrants$applications

# Bar plot for success rate by discipline and gender
ggplot(NWOGrants, aes(x = factor(discipline), y = success_rate, fill = factor(gender))) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  scale_fill_manual(values = c("navyblue", "pink"), labels = c("Male", "Female")) +
  labs(title = "Success Rate by Discipline and Gender",
       x = "Discipline", y = "Success Rate", fill = "Gender") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Model 1

model_formula <- alist(
  awards ~ binomial(applications, p),
  logit(p) <- a[gender],
  a[gender] ~ dnorm(0, 0.5)
)

model <- ulam(
  model_formula,
  data = NWOGrants,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model, depth=2)



post_samples <- extract.samples(model)
p_samples_m <- inv_logit(post_samples$a[,1])  # For male
p_samples_f <- inv_logit(post_samples$a[,2])  # For female

differences <- post_samples$a[,1] - post_samples$a[,2]
precis(differences)

# Create data frames for posterior samples of male and female probabilities
data_m <- data.frame(prob = p_samples_m, gender = 'Male')
data_f <- data.frame(prob = p_samples_f, gender = 'Female')

# Combine the data into one data frame
posterior_data <- rbind(data_m, data_f)

# Create a density plot of the posterior probabilities
ggplot(posterior_data, aes(x=prob, fill=gender)) +
  geom_density(alpha=0.5) + 
  theme_minimal() +  
  labs(title="Posterior Distribution of Probability (p)", 
       x="Probability", 
       y="Density") + 
  scale_fill_manual(values = c("blue", "pink"))  




# Model 2

model_formula2 <- alist(
  awards ~ binomial(applications, p),
  logit(p) <- a[gender] + b[discipline],
  a[gender] ~ dnorm(0, 0.5),
  b[discipline] ~ dnorm(0, 0.5)
)

model2 <- ulam(
  model_formula2,
  data = NWOGrants,
  chains = 4, cores = 4, log_lik=TRUE
)

precis(model2, depth=2)




# Extract posterior samples for a[gender] and b[discipline]
post_samples2 <- extract.samples(model2)

differences_2 <- post_samples2$a[,1] - post_samples2$a[,2]
precis(differences_2)

num_disciplines <- length(unique(NWOGrants$discipline))

posterior_data2 <- data.frame()


# Loop through disciplines to calculate probabilities for male and female
for(d in 1:num_disciplines) {
  # For males in this discipline
  p_samples_m <- inv_logit(post_samples2$a[,1] + post_samples2$b[,d])  # Male
  data_m <- data.frame(prob = p_samples_m, gender = 'Male', discipline = factor(d))
  
  # For females in this discipline
  p_samples_f <- inv_logit(post_samples2$a[,2] + post_samples2$b[,d])  # Female
  data_f <- data.frame(prob = p_samples_f, gender = 'Female', discipline = factor(d))
  
  # Combine male and female data for this discipline
  posterior_data2 <- rbind(posterior_data2, data_m, data_f)
}

# Ensure that discipline is a factor
posterior_data2$discipline <- factor(posterior_data2$discipline, 
                                     levels = 1:num_disciplines,
                                     labels = unique(NWOGrants$discipline))

# Plot the posterior distributions of probabilities by gender and discipline
ggplot(posterior_data2, aes(x=prob, fill=gender)) +
  geom_density(alpha=0.5) +  
  theme_minimal() +  
  labs(title="Posterior Distribution of Probability (p) by Gender and Discipline", 
       x="Probability", 
       y="Density") + 
  scale_fill_manual(values = c("blue", "pink")) + 
  facet_wrap(~discipline)  




# Problem 4 ===================================================================#


N <- 100
G <-  rbinom(N, 1, 0.5)   # Assume 1=male and 0=female
CS <- runif(N, 0, 1)
D <-  rbinom(N, 1, plogis(0.25*G + CS)) 
A <-  rbinom(N, 1, plogis(D - CS + G)) 

model1 <- ulam(
  alist(
    A ~ bernoulli(p),
    logit(p) <- alpha + beta_D * D + beta_G * G,
    alpha ~ normal(0, 1), 
    beta_D ~ normal(0, 1), 
    beta_G ~ normal(0, 1)
  ), data = list(G = G, CS = CS, D = D, A = A), chains = 4, cores = 4
)

precis(model1)



# Problem 5 ===================================================================#


data(Primates301)
print(Primates301)

Primates301 <- na.omit(Primates301[, c("social_learning", "brain", "research_effort")])
Primates301$brain = standardize(log(Primates301$brain))

# 5.a) ===========================#

model1 <- ulam(
  alist(
    social_learning ~ poisson(lambda),
    log(lambda) <- a + b * brain,
    a ~ normal(0,1),
    b ~ normal(0,1)
  ), data=Primates301, chains=4, cores=4, log_lik=TRUE
)

precis(model1)

# Extract posterior samples
post <- extract.samples(model1)

# Create a sequence of brain sizes for prediction
brain_seq <- seq(from = min(Primates301$brain), to = max(Primates301$brain), length.out = 100)

# Compute the mean and credible intervals of the predicted lambda
lambda_pred <- sapply(brain_seq, function(brain) {
  lambda <- exp(post$a + post$b * brain)
  c(mean = mean(lambda), PI(lambda, prob = 0.89))
})

# Convert to a data frame for ggplot
pred_df <- data.frame(
  brain = brain_seq,
  mean = lambda_pred[1, ],
  lower = lambda_pred[2, ],
  upper = lambda_pred[3, ]
)

ggplot() +
  geom_point(data = Primates301, aes(x = brain, y = social_learning), color = "black", alpha = 0.6) +
  geom_line(data = pred_df, aes(x = brain, y = mean), color = "red", size = 1) +
  geom_ribbon(data = pred_df, aes(x = brain, ymin = lower, ymax = upper), alpha = 0.2, fill = "red") +
  labs(
    title = "Posterior Predictions of Social Learning",
    x = "Standardized Log(Brain Size)",
    y = "Social Learning"
  ) +
  theme_minimal()



# 5.b) ===========================#

Primates301$research_effort = log(Primates301$research_effort)

model2 <- ulam(
  alist(
    social_learning ~ poisson(lambda),
    log(lambda) <- a + b * brain + c * research_effort,
    a ~ normal(0,1),
    b ~ normal(0,1),
    c ~ normal(0,1)
  ), 
  data = Primates301, chains = 4, cores = 4, log_lik = TRUE
)

precis(model2)



# Extract posterior samples from the extended model
post2 <- extract.samples(model2)

# Create a grid of predictor values
brain_seq <- seq(from = min(Primates301$brain), to = max(Primates301$brain), length.out = 50)
research_seq <- seq(from = min(Primates301$research_effort), to = max(Primates301$research_effort), length.out = 50)

# Generate predictions over the grid
predictions <- expand.grid(brain = brain_seq, research_effort = research_seq)
predictions$lambda_mean <- NA
predictions$lambda_lower <- NA
predictions$lambda_upper <- NA

# Loop through the grid to calculate posterior predictions
for (i in 1:nrow(predictions)) {
  lambda <- exp(post2$a + post2$b * predictions$brain[i] + post2$c * predictions$research_effort[i])
  predictions$lambda_mean[i] <- mean(lambda)
  predictions$lambda_lower[i] <- PI(lambda, prob = 0.89)[1]
  predictions$lambda_upper[i] <- PI(lambda, prob = 0.89)[2]
}

ggplot() +
  geom_line(data = predictions, aes(x = brain, y = lambda_mean, group = research_effort, color = research_effort), size = 1) +
  geom_ribbon(data = predictions, aes(x = brain, ymin = lambda_lower, ymax = lambda_upper, group = research_effort, fill = research_effort), alpha = 0.2) +
  geom_point(data = Primates301, aes(x = brain, y = social_learning), color = "black", alpha = 0.6) +
  scale_color_gradientn(colors = c("green", "lightblue","steelblue", "darkblue"), limits = c(4, 7)) +
  scale_fill_gradientn(colors = c("green", "lightblue","steelblue", "darkblue"), limits = c(4, 7)) +
  labs(
    title = "Posterior Predictions of Social Learning (Extended Model)",
    x = "Standardized Log(Brain Size)",
    y = "Social Learning",
    color = "Research Effort",
    fill = "Research Effort"
  ) +
  theme_minimal()

compare(model1, model2)


