# ===================================================================== #
#                 END-TO-END DATA SCIENCE PROJECT IN R                  #
#                 Domain: Finance | Loan Default Prediction             #
# ===================================================================== #

# -------------------------- Load Libraries ---------------------------- #
library(tidyverse)
library(janitor)
library(skimr)
library(tidymodels)
library(themis)
library(vip)
library(pROC)
library(corrplot)
library(rsample)
library(recipes)
library(parsnip)
library(workflows)
library(yardstick)
library(ranger)

set.seed(123)

# ------------------------ 1️⃣ Problem Definition ------------------------ #
# Goal: Predict whether a loan application will be approved (Y) or rejected (N)

# ------------------------ 2️⃣ Generate Dataset -------------------------- #
df <- data.frame(
  Gender = sample(c("Male", "Female"), 500, replace = TRUE),
  Married = sample(c("Yes", "No"), 500, replace = TRUE),
  ApplicantIncome = rnorm(500, mean = 4000, sd = 1500),
  LoanAmount = rnorm(500, mean = 150, sd = 50),
  Credit_History = sample(c(1, 0), 500, replace = TRUE, prob = c(0.8, 0.2)),
  Loan_Status = sample(c("Y", "N"), 500, replace = TRUE, prob = c(0.7, 0.3))
)

# Create folders if not exist
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("outputs")) dir.create("outputs")
if (!dir.exists("reports")) dir.create("reports")
if (!dir.exists("models")) dir.create("models")
if (!dir.exists("scripts")) dir.create("scripts")

write.csv(df, "data/loan_data.csv", row.names = FALSE)
head(df)

# ------------------------ 3️⃣ Load and Clean Data ------------------------ #
loan_data <- read_csv("data/loan_data.csv") %>%
  clean_names()

glimpse(loan_data)
skim(loan_data)

# Remove duplicates and handle missing values
loan_data <- loan_data %>%
  distinct() %>%
  mutate(
    gender = replace_na(gender, "Male"),
    married = replace_na(married, "No"),
    applicant_income = replace_na(applicant_income, median(applicant_income, na.rm = TRUE)),
    loan_amount = replace_na(loan_amount, median(loan_amount, na.rm = TRUE)),
    credit_history = replace_na(credit_history, 1),
    loan_status = ifelse(loan_status == "Y", 0, 1),  # 0 = Approved, 1 = Rejected
    loan_status = as.factor(loan_status)
  )

# Feature engineering
loan_data <- loan_data %>%
  mutate(
    loan_to_income = loan_amount / applicant_income,
    gender = as.factor(gender),
    married = as.factor(married),
    credit_history = as.factor(credit_history)
  )

skim(loan_data)

# ------------------------ 4️⃣ Exploratory Data Analysis ------------------ #

# Save correlation plot
numeric_vars <- loan_data %>% select(applicant_income, loan_amount, loan_to_income)
png("outputs/correlation_plot.png", width = 800, height = 600)
corrplot(cor(numeric_vars), method = "color")
dev.off()

# Loan Status Distribution
p1 <- ggplot(loan_data, aes(loan_status, fill = loan_status)) +
  geom_bar() +
  labs(title = "Loan Status Distribution", x = "Loan Status", y = "Count")
ggsave("outputs/loan_status_distribution.png", p1, width = 6, height = 4)

# Loan Amount vs Applicant Income
p2 <- ggplot(loan_data, aes(applicant_income, loan_amount, color = loan_status)) +
  geom_point(alpha = 0.7) +
  labs(title = "Loan Amount vs Applicant Income")
ggsave("outputs/loan_amount_vs_income.png", p2, width = 6, height = 4)

# Income distribution by approval status
p3 <- ggplot(loan_data, aes(x = applicant_income, fill = loan_status)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  labs(title = "Applicant Income Distribution by Loan Status")
ggsave("outputs/income_distribution_by_status.png", p3, width = 6, height = 4)

# Credit History vs Default Rate
p4 <- loan_data %>%
  group_by(credit_history) %>%
  summarise(default_rate = mean(as.numeric(loan_status) == 2)) %>%
  ggplot(aes(x = credit_history, y = default_rate)) +
  geom_col(fill = "steelblue") +
  labs(title = "Default Rate by Credit History", x = "Credit History", y = "Default Rate")
ggsave("outputs/default_rate_by_credit_history.png", p4, width = 6, height = 4)

# ------------------------ 5️⃣ Train-Test Split --------------------------- #
set.seed(123)
data_split <- initial_split(loan_data, prop = 0.8, strata = loan_status)
train_data <- training(data_split)
test_data  <- testing(data_split)

write.csv(train_data, "data/train_data.csv", row.names = FALSE)
write.csv(test_data, "data/test_data.csv", row.names = FALSE)

# ------------------------ 6️⃣ Data Preprocessing -------------------------- #
rec <- recipe(loan_status ~ gender + married + applicant_income + loan_amount + credit_history + loan_to_income,
              data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ------------------------ 7️⃣ Logistic Regression ------------------------ #
log_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

wf_log <- workflow() %>%
  add_model(log_model) %>%
  add_recipe(rec)

fit_log <- fit(wf_log, data = train_data)

# ------------------------ 8️⃣ Random Forest ------------------------------ #
rf_model <- rand_forest(mtry = 3, trees = 500, min_n = 5) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf_rf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rec)

fit_rf <- fit(wf_rf, data = train_data)

# ------------------------ 9️⃣ Model Evaluation --------------------------- #
# Logistic Regression
log_preds <- predict(fit_log, test_data, type = "prob") %>%
  bind_cols(test_data %>% select(loan_status))
roc_auc_log <- roc_auc(log_preds, truth = loan_status, .pred_1)

log_class <- predict(fit_log, test_data)
log_results <- bind_cols(log_class, test_data %>% select(loan_status))
log_metrics <- metrics(log_results, truth = loan_status, estimate = .pred_class)
log_conf <- conf_mat(log_results, truth = loan_status, estimate = .pred_class)

# Random Forest
rf_preds <- predict(fit_rf, test_data, type = "prob") %>%
  bind_cols(test_data %>% select(loan_status))
roc_auc_rf <- roc_auc(rf_preds, truth = loan_status, .pred_1)

rf_class <- predict(fit_rf, test_data)
rf_results <- bind_cols(rf_class, test_data %>% select(loan_status))
rf_metrics <- metrics(rf_results, truth = loan_status, estimate = .pred_class)
rf_conf <- conf_mat(rf_results, truth = loan_status, estimate = .pred_class)

# Print results
print("📊 Logistic Regression Results:")
print(log_metrics)
print(roc_auc_log)

print("🌲 Random Forest Results:")
print(rf_metrics)
print(roc_auc_rf)

# ------------------------ 🔟 Feature Importance -------------------------- #
final_rf <- extract_fit_parsnip(fit_rf)$fit
importance_vec <- final_rf$variable.importance
importance_df <- data.frame(
  Feature = names(importance_vec),
  Importance = as.numeric(importance_vec)
) %>%
  arrange(desc(Importance))

write.csv(importance_df, "outputs/feature_importance.csv", row.names = FALSE)

p5 <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Feature)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Feature Importance - Random Forest Model", x = "Feature", y = "Importance") +
  theme_minimal()
ggsave("outputs/feature_importance_plot.png", p5, width = 6, height = 4)

# ------------------------ 1️⃣1️⃣ Save Model ------------------------------ #
saveRDS(fit_rf, "outputs/random_forest_loan_model.rds")

# ===================================================================== #
# ✅ END OF PROJECT
# ===================================================================== #