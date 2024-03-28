require("limma")

design_matrix <- read.table("/home/maceor22/.cache/RNAlysis/2024_03_20/design_mat_22.csv", header=TRUE, sep= ",")
condition <- factor(design_matrix$condition, levels=c("control", "defeated"))
group <- factor(design_matrix$group, levels=c("mPFC", "vHPC"))

design <- model.matrix(~ condition + group)
colnames(design)[1] <- "Intercept"

count_data <- read.table("/home/maceor22/.cache/RNAlysis/2024_03_20/reads_CPM.csv", header=TRUE, sep= ",", row.names = 1)
voom_object <- voom(count_data, design, plot=FALSE, save.plot=TRUE)

fit <- lmFit(voom_object, design)

contrast <- makeContrasts("conditiondefeated", levels = design)
contrast_fit <- contrasts.fit(fit, contrast)
contrast_bayes <- eBayes(contrast_fit)
res <- topTable(contrast_bayes, n=Inf)
res_ordered <- res[order(res$adj.P.Val),]
write.csv(as.data.frame(res_ordered),file="/home/maceor22/.cache/RNAlysis/2024_03_20/e129dc41470cd0231769803091e184a22fae1c7c/LimmaVoom_condition_defeated_vs_control.csv")
