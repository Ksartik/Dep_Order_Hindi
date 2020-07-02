library(ggplot2)
library(lme4)
library(reshape2)
library(dplyr)

alpha <- 0.05

lci <- function (x, n) {
  return (mean(x) - qt(1- alpha/2, (n - 1))*sd(x)/sqrt(n))
}

uci <- function (x, n) {
  return (mean(x) + qt(1- alpha/2, (n - 1))*sd(x)/sqrt(n))
}

folder <- "order_param_comparison"
'%ni%' <- Negate('%in%')

argts <- c("k1", "k1s", "k2", "k4")
subjs <- c("k1", "k1s")
objs <- c("k2", "k4")

# 
# Argument - adjunct
df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel %in% argts) & (d2_deprel %ni% argts)) | 
                    ((d1_deprel %ni% argts) & (d2_deprel %in% argts)))
df$dep_order <- ifelse(((df$d1_deprel %in% argts) & 
                        (df$d2_deprel %ni% argts)), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

model <- glm (dep_order ~ hdmi_diff + accessibility_diff + cosdist_diff + case_diff,
              data=df,
              family="binomial")

model %>% summary()

# 
# Subject - objects
df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel %in% subjs) & (d2_deprel %in% objs)) | 
                    ((d1_deprel %in% objs) & (d2_deprel %in% subjs)))
df$dep_order <- ifelse(((df$d1_deprel %in% subjs) & 
                        (df$d2_deprel %in% objs)), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

model <- glm (dep_order ~ hdmi_diff + accessibility_diff + cosdist_diff + case_diff,
              data=df,
              family="binomial")

model %>% summary()

# 
# Direct objects - Indirect objects
df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel == "k2") & (d2_deprel == "k4")) | 
                    ((d1_deprel == "k4") & (d2_deprel == "k2")))
df$dep_order <- ifelse(((df$d1_deprel == "k2") & 
                        (df$d2_deprel == "k4")), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

model <- glm (dep_order ~ hdmi_diff + accessibility_diff + cosdist_diff + case_diff,
              data=df,
              family="binomial")

model %>% summary()

df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel %in% argts) & (d2_deprel %ni% argts)) | 
                    ((d1_deprel %ni% argts) & (d2_deprel %in% argts)))
df$dep_order <- ifelse(((df$d1_deprel %in% argts) & 
                        (df$d2_deprel %ni% argts)), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

summ_df <- df %>% group_by(dep_order) %>% 
  summarise(hdmi_diff_mean=mean(hdmi_diff), hdmi_diff_lci=lci(hdmi_diff, n()), hdmi_diff_uci=uci(hdmi_diff, n()),
            accessibility_diff_mean=mean(accessibility_diff), accessibility_diff_lci=lci(accessibility_diff, n()), accessibility_diff_uci=uci(accessibility_diff, n()),
            cosdist_diff_mean=mean(cosdist_diff), cosdist_diff_lci=lci(cosdist_diff, n()), cosdist_diff_uci=uci(cosdist_diff, n()),
            case_diff_mean=mean(case_diff), case_diff_lci=lci(case_diff, n()), case_diff_uci=uci(case_diff, n())) %>%
  mutate(word_order="Arg--Adj")
# 
# 
df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel %in% subjs) & (d2_deprel %in% objs)) | 
                    ((d1_deprel %in% objs) & (d2_deprel %in% subjs)))
df$dep_order <- ifelse(((df$d1_deprel %in% subjs) & 
                        (df$d2_deprel %in% objs)), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

summ_df <- rbind(summ_df, 
  df %>% group_by(dep_order) %>% 
    summarise(hdmi_diff_mean=mean(hdmi_diff), hdmi_diff_lci=lci(hdmi_diff, n()), hdmi_diff_uci=uci(hdmi_diff, n()),
              accessibility_diff_mean=mean(accessibility_diff), accessibility_diff_lci=lci(accessibility_diff, n()), accessibility_diff_uci=uci(accessibility_diff, n()),
              cosdist_diff_mean=mean(cosdist_diff), cosdist_diff_lci=lci(cosdist_diff, n()), cosdist_diff_uci=uci(cosdist_diff, n()),
              case_diff_mean=mean(case_diff), case_diff_lci=lci(case_diff, n()), case_diff_uci=uci(case_diff, n())) %>%
    mutate(word_order="S--O"))
# 
# 
df <- read.csv('all_order_params.csv')

df <- df %>% filter(((d1_deprel == "k2") & (d2_deprel == "k4")) | 
                    ((d1_deprel == "k4") & (d2_deprel == "k2")))
df$dep_order <- ifelse(((df$d1_deprel == "k2") & 
                        (df$d2_deprel == "k4")), 1, 0)
df$accessibility_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_accessibility - df$d2_accessibility,
                                df$d2_accessibility - df$d1_accessibility))
df$hdmi_diff <- scale(ifelse(df$dep_order == 1, 
                            df$d1_hdmi - df$d2_hdmi,
                            df$d2_hdmi - df$d1_hdmi))
df$cosdist_diff <- scale(ifelse(df$dep_order == 1, 
                              df$d1_cosdist - df$d2_cosdist,
                              df$d2_cosdist - df$d1_cosdist))
df$case_diff <- scale(ifelse(df$dep_order == 1, 
                                df$d1_case - df$d2_case,
                                df$d2_case - df$d1_case))

summ_df <- rbind(summ_df, 
  df %>% group_by(dep_order) %>% 
    summarise(hdmi_diff_mean=mean(hdmi_diff), hdmi_diff_lci=lci(hdmi_diff, n()), hdmi_diff_uci=uci(hdmi_diff, n()),
              accessibility_diff_mean=mean(accessibility_diff), accessibility_diff_lci=lci(accessibility_diff, n()), accessibility_diff_uci=uci(accessibility_diff, n()),
              cosdist_diff_mean=mean(cosdist_diff), cosdist_diff_lci=lci(cosdist_diff, n()), cosdist_diff_uci=uci(cosdist_diff, n()),
              case_diff_mean=mean(case_diff), case_diff_lci=lci(case_diff, n()), case_diff_uci=uci(case_diff, n())) %>%
    mutate(word_order="DO--IO"))

summ_df %>% 
  mutate(dep_order_xy = ifelse(dep_order == 1, "Order = X,Y", "Order = Y,X")) %>%
  ggplot(mapping = aes(x=dep_order_xy, y=hdmi_diff_mean, color=word_order, group=word_order)) + geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = hdmi_diff_lci, ymax = hdmi_diff_uci), position = position_dodge(0.01)) + 
  labs(x="Word-order", y = "HDMI(X) - HDMI(Y)", color="X,Y") + 
  scale_color_manual(labels = c("Arg,Adj", "DO,IO", "S,O"), 
                     values = c("red", "green", "blue")) +
  theme(
    legend.position = c(0.95, 0.5),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2),
    # legend.title=element_blank(),
    axis.title.x=element_blank()
  ) + 
  ggsave(paste(sep="/", folder, "all_orders_hdmi.png"), width = 3.5, height = 3.5, dpi = 300, units = "in")


summ_df %>% 
  mutate(dep_order_xy = ifelse(dep_order == 1, "Order = X,Y", "Order = Y,X")) %>%
  ggplot(mapping = aes(x=dep_order_xy, y=accessibility_diff_mean, color=word_order, group=word_order)) + geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = accessibility_diff_lci, ymax = accessibility_diff_uci), position = position_dodge(0.01)) + 
  labs(x = "Word-order", y = "Animacy(X) - Animacy(Y)", color="X,Y") + 
  scale_color_manual(labels = c("Arg,Adj", "DO,IO", "S,O"), 
                     values = c("red", "green", "blue")) +
  theme(
    legend.position = c(0.3, 0.5),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2),
    # legend.title=element_blank(),
    axis.title.x=element_blank()
  ) + 
  ggsave(paste(sep="/", folder, "all_orders_animacy.png"), width = 3.5, height = 3.5, dpi = 300, units = "in")

summ_df %>% 
  mutate(dep_order_xy = ifelse(dep_order == 1, "Order = X,Y", "Order = Y,X")) %>%
  ggplot(mapping = aes(x=dep_order_xy, y=cosdist_diff_mean, color=word_order, group=word_order)) + geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = cosdist_diff_lci, ymax = cosdist_diff_uci), position = position_dodge(0.01)) + 
  labs(x = "Word-order", y = "Similarity(X) - Similarity(Y)", color="X,Y") + 
  scale_color_manual(labels = c("Arg,Adj", "DO,IO", "S,O"), 
                     values = c("red", "green", "blue")) +
  theme(
    legend.position = c(0.4, 0.95),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2),
    # legend.title=element_blank(),
    axis.title.x=element_blank()
  ) + 
  ggsave(paste(sep="/", folder, "all_orders_sim.png"), width = 3.5, height = 3.5, dpi = 300, units = "in")

summ_df %>% 
  mutate(dep_order_xy = ifelse(dep_order == 1, "Order = X,Y", "Order = Y,X")) %>%
  ggplot(mapping = aes(x=dep_order_xy, y=case_diff_mean, color=word_order, group=word_order)) + geom_point() + geom_line() + 
  geom_errorbar(aes(ymin = case_diff_lci, ymax = case_diff_uci), position = position_dodge(0.01)) + 
  labs(x = "Word-order", y = "Case(X) - Case(Y)", color="X,Y") + 
  scale_color_manual(labels = c("Arg,Adj", "DO,IO", "S,O"), 
                     values = c("red", "green", "blue")) +
  theme(
    legend.position = c(0.3, 0.4),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(2, 2, 2, 2),
    # legend.title=element_blank(),
    axis.title.x=element_blank()
  ) + 
  ggsave(paste(sep="/", folder, "all_orders_case.png"), width = 3.5, height = 3.5, dpi = 300, units = "in")
