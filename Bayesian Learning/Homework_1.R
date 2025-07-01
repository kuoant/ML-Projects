#== Problem 1 =================================================================#


library(ggplot2)
library(parallel)

# Simulation function
sim_function <- function(N = 100) {
  M <- rnorm(N)
  E <- rnorm(N, mean = M)
  CI <- rnorm(N, mean = E + M)
  WL <- rnorm(N, mean = E + CI)
  
  c(
    coef(lm(WL ~ E))["E"],           # Biased
    coef(lm(WL ~ E + CI))["E"],      # Unbiased
    coef(lm(WL ~ E + CI + M))["E"]   # Fully adjusted
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Unbiased", "Fully Adjusted"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.8) +  
  theme_minimal() +
  labs(title = "Effect of E on WL", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen")) 





#== Problem 2 =================================================================#



# Simulation function
sim_function <- function(N = 100) {
  CE <- rnorm(N)
  EL <- rnorm(N, mean = CE)
  F <- rnorm(N, mean = EL)
  JS <- rnorm(N, mean = EL + F + CE)
  
  c(
    coef(lm(JS ~ EL))["EL"],
    coef(lm(JS ~ EL + CE))["EL"],
    coef(lm(JS ~ EL + F + CE))["EL"]
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Unbiased", "Fully Adjusted"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.8) +  
  theme_minimal() +
  labs(title = "Effect of Education Level on Job Satisfaction", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen"))




#== Problem 3 =================================================================#


# Simulation function
sim_function <- function(N = 100) {
  A <- rnorm(N)
  SI <- rnorm(N, mean = A)
  M <- rnorm(N, mean = SI)
  RR <- rnorm(N, mean = M + SI + A)
  
  c(
    coef(lm(RR ~ M))["M"],
    coef(lm(RR ~ M + SI))["M"],
    coef(lm(RR ~ M + SI + A))["M"]
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Unbiased", "Fully Adjusted"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.5) +  
  theme_minimal() +
  labs(title = "Effect of Medication on Recovery Rate", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen"))



#== Problem 4 =================================================================#


# Simulation function
sim_function <- function(N = 100) {
  EC <- rnorm(N)
  Q <- rnorm(N)
  B <- rnorm(N, mean = EC)
  A <- rnorm(N, mean = Q + B)
  S <- rnorm(N, mean = A + EC + Q)
  
  c(
    coef(lm(S ~ A))["A"],
    coef(lm(S ~ A + Q + EC))["A"],
    coef(lm(S ~ A + Q + B))["A"]
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Unbiased using EC", "Unbiased using B"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.5) +  
  theme_minimal() +
  labs(title = "Effect of Advertising on Sales", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen"))



#== Problem 5 =================================================================#


# Simulation function
sim_function <- function(N = 100) {
  OSN <- rnorm(N)
  PI <- rnorm(N)
  SM <- rnorm(N, mean = OSN + PI)
  MH <- rnorm(N, mean = SM + OSN)
  
  c(
    coef(lm(MH ~ SM))["SM"],
    coef(lm(MH ~ SM + OSN))["SM"],
    coef(lm(MH ~ SM + OSN + PI))["SM"]
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Correct", "Including PI"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.5) +  
  theme_minimal() +
  labs(title = "Effect of Social Media Use on Mental Health", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen"))



#== Problem 6 =================================================================#


# Simulation function
sim_function <- function(N = 100) {
  D <- rnorm(N)
  PE <- rnorm(N, mean = D)
  HC <- rnorm(N, mean = PE + D)
  US <- rnorm(N, mean = HC)
  
  c(
    coef(lm(HC ~ PE))["PE"],
    coef(lm(HC ~ PE + D))["PE"],
    coef(lm(HC ~ PE + D + US))["PE"]
  )
}

# Run Simulation
results <- do.call(rbind, mclapply(1:10000, function(x) sim_function(), mc.cores = 4))

# Conversion
df <- data.frame(
  Effect = c(results[, 1], results[, 2], results[, 3]),
  Model = rep(c("Biased", "Correct", "Including US"), each = 10000)
)

# Plot
ggplot(df, aes(x = Effect, fill = Model)) +
  geom_density(alpha = 0.5) +  
  theme_minimal() +
  labs(title = "Effect of Pesticide Exposure on Health Condition", x = "Estimated Effect", y = "Density") +
  scale_fill_manual(values = c("lightcoral", "lightblue", "lightgreen"))


