library(ggplot2)
library(lme4)
library(reshape2)
library(dplyr)

folder <- "dist_param_comparison"
'%ni%' <- Negate('%in%')

alpha <- 0.05

lci <- function (x, n) {
  return (mean(x) - qt(1- alpha/2, (n - 1))*sd(x)/sqrt(n))
}

uci <- function (x, n) {
  return (mean(x) + qt(1- alpha/2, (n - 1))*sd(x)/sqrt(n))
}

df <- read.csv('dist_params.csv', header=FALSE)

names(df) <- c("sent_id", "sent_pos", "dep_dist", "case", "raw_deprel", "deprel", "animacy", "cosine_dist", "hdmi")
coreargs <- c("sub", "dobj", "iobj")

df$dep_type <- ifelse(df$raw_deprel %in% coreargs, "arg", "adj")

df$case <- as.factor(df$case)
df$animacy <- as.factor(df$animacy)
df$sent_id <- as.factor(df$sent_id)

df <- df %>% filter (!((deprel == "iobj") & (case == 0)))

df <- df %>% filter(dep_dist > 1)
df$dep_dist <- scale(df$dep_dist)
df$cosine_dist <- scale(df$cosine_dist)
df$hdmi <- scale(df$hdmi)

model <- lmer(formula=dep_dist ~ animacy + cosine_dist + hdmi + case, 
              data = df %>% filter(deprel == "adj"))
model %>% summary()

model <- lmer(formula=dep_dist ~ animacy + cosine_dist + hdmi + case, 
              data = df %>% filter(deprel == "sub"))
model %>% summary()

model <- lmer(formula=dep_dist ~ animacy + cosine_dist + hdmi + case, 
              data = df %>% filter(deprel == "dobj"))
model %>% summary()

model <- lmer(formula=dep_dist ~ animacy + cosine_dist + hdmi, 
              data = df %>% filter(deprel == "iobj"))
model %>% summary()

df %>% filter(dep_dist > 1) %>% group_by(dep_dist, deprel) %>% 
  summarise(n = n()) %>% 
  mutate(ln_dist = log(dep_dist), ln_freq=log(n)) %>%
  ggplot(mapping = aes(x=ln_dist, y=ln_freq, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "ln(Dependency distance)", y = "ln(frequency)", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "Direct Objects", "Indirect Objects", "Subjects"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.45, .45),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2)
  ) + 
  # geom_errorbar(aes(ymin = lci_hdmi, ymax = uci_hdmi), position = "dodge") #+ 
  ggsave(paste(sep="/", folder, "deps_freq_dist.png"), width = 4.4, height = 4.1, dpi = 300, units = "in")

df %>% filter(dep_dist > 1) %>% group_by(dep_dist, deprel) %>% 
  summarise(mean_hdmi=mean(hdmi), uci_hdmi=uci(hdmi, n()), lci_hdmi=lci(hdmi, n())) %>% 
  ggplot(mapping = aes(x=dep_dist, y=mean_hdmi, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "Dependency distance", y = "HDMI", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "Direct Objects", "Indirect objects", "Subjects"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2)
  ) + 
  geom_errorbar(aes(ymin = lci_hdmi, ymax = uci_hdmi), position = "dodge") #+ 
  ggsave(paste(sep="/", folder, "deps_hdmi_dist.png"), width = 4.4, height = 4.1, dpi = 300, units = "in")

df %>% filter(dep_dist > 1) %>% group_by(dep_dist, deprel) %>% 
  summarise(mean_sim=mean(cosine_dist), uci_sim=uci(cosine_dist, n()), lci_sim=lci(cosine_dist, n())) %>% 
  ggplot(mapping = aes(x=dep_dist, y=mean_sim, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "Dependency distance", y = "Similarity", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "Direct Objects", "Indirect objects", "Subjects"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.95, .45),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2)
  ) + 
  geom_errorbar(aes(ymin = lci_sim, ymax = uci_sim), position = position_dodge(0.2)) + 
  ggsave(paste(sep="/", folder, "deps_sim_dist_g1_ci.png"), width = 4.4, height = 4.1, dpi = 300, units = "in")

df %>% filter(dep_dist > 1) %>% group_by(dep_dist, deprel, animacy) %>% 
  summarise(n_animate=n()) %>% group_by(dep_dist, deprel) %>%
  mutate(t_animate= sum(n_animate)) %>% group_by(dep_dist, deprel, animacy) %>% 
  mutate(animate_prop=round(n_animate/t_animate,2)) %>% filter(animacy == 1) %>%
  mutate(lci_animate_prop=animate_prop-qnorm(1-alpha)*sqrt((1/t_animate)*animate_prop*(1-animate_prop)),
         uci_animate_prop=animate_prop+qnorm(1-alpha)*sqrt((1/t_animate)*animate_prop*(1-animate_prop))) %>%
  ggplot(mapping = aes(x=dep_dist, y=animate_prop, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "Dependency distance", y = "Proportion of human-animate dependents", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "DO", "IO", "S"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.38, .67),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2),
  ) + 
  guides(col=guide_legend(nrow=2,byrow=TRUE)) +
  geom_errorbar(aes(ymin = lci_animate_prop, ymax = uci_animate_prop), position = position_dodge(0.2)) + 
  ggsave(paste(sep="/", folder, "deps_animacy_dist_ci_g1.png"), width = 4.4, height = 4.1, dpi = 400, units = "in")
 

df %>% filter(dep_dist > 1) %>% group_by(dep_dist, deprel, case) %>% 
  summarise(n_case=n()) %>% group_by(dep_dist, deprel) %>%
  mutate(t_case= sum(n_case)) %>% group_by(dep_dist, deprel, case) %>% 
  mutate(case_prop=round(n_case/t_case,2)) %>% filter(case == 1) %>%
  mutate(lci_case_prop=case_prop-qnorm(1-alpha)*sqrt((1/t_case)*case_prop*(1-case_prop)),
         uci_case_prop=case_prop+qnorm(1-alpha)*sqrt((1/t_case)*case_prop*(1-case_prop))) %>%
  ggplot(mapping = aes(x=dep_dist, y=case_prop, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "Dependency distance", y = "Proportion of case-marked dependents", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "Direct Objects", "Indirect objects", "Subjects"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.95, .65),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2)
  ) + 
  geom_errorbar(aes(ymin = lci_case_prop, ymax = uci_case_prop), position = position_dodge(0.2)) + 
  ggsave(paste(sep="/", folder, "deps_case_dist_ci_g1.png"), width = 4.4, height = 4.1, dpi = 400, units = "in")


df %>% filter(dep_dist > 1) %>% 
  group_by(dep_dist, deprel) %>% 
  summarise(n=n(), mean_sentpos=mean(sent_pos), lci_sentpos=lci(sent_pos, n()), uci_sentpos=uci(sent_pos, n())) %>% 
  ggplot(mapping = aes(x=dep_dist, y=mean_sentpos, col=deprel)) + geom_point() + geom_smooth(se=FALSE) + 
  labs(x = "Dependency distance", y = "Mean sentence position", color="Dependency type") +
  scale_color_manual(labels = c("Adjuncts", "Direct objects", "Indirect objects", "Subjects"), 
                     values = c("saddlebrown", "blue", "limegreen", "red")) +
  theme(
    legend.position = c(.95, .95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2)
  ) + 
  geom_errorbar(aes(ymin = lci_sentpos, ymax = uci_sentpos), position = position_dodge(0.2)) + 
  ggsave(paste(sep="/", folder, "deps_sentpos_dist_ci_g1.png"), width = 4.4, height = 4.1, dpi = 400, units = "in")

