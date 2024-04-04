################################################################################
###                    NLP RUG 2024 - Universal Inflector                    ###
### 2A: Maria FRANCIS, Vitalii HIRAK, Valentine LUCQUIAULT, Frieso TURKSTRA ###
################################################################################

pacman:: p_load(car, readr, tidytext, dplyr,scales, ggplot2, stringr, textdata, 
                tidyr, tidylo, tidyverse, rstatix)

#Importing data
#all the data we have
fulldata <- read.csv("fulldata.csv", sep=";")
#factorising the language families to allow for group comparisons in ANOVA
fulldata$family <- as.factor(fulldata$family)

#scores obtained by the shared task, by ByT5 when trained on the features, 
#and by ByT5 when not trained on the features
#used to do check homoscedasticity for group comparisons (Levene's test)
scores_permodel <- read.csv("scores_permodel.csv", sep=";")

################################################################################
###################          Comparing performances          ###################
################################################################################


#Shared task accuracy x ByT5 accuracy, colouring by language family
ggplot(fulldata, aes(x=sharedtask, y=byt5, colour=family))+
  geom_text(aes(label = wals_code), size=8)+xlab("Shared Task Accuracy")+ylab("Bytes Accuracy")+
  theme(legend.text = element_text(size = 10)) + labs(color="Language Family")

#correlation between our model and the shared task
cor.test(fulldata$sharedtask, fulldata$byt5, method="pearson")
#p-value = 0.0001157, R²=0.85 => SIGNIFICANT CORRELATION



#is there a significant difference between our model and the shared task?

#checking assumptions of normality and homoscedasticity
#checking for groups byt5_wfeat and sharedtask only, so lines 1 to 28
leveneTest(score~type, data=scores_permodel[1:28,]) #p-value>0.05 so not homoscedastic
shapiro.test(scores_permodel$score[15:28]) #normality for ByT5 with features: p-value>0.05 so normal 
shapiro.test(scores_permodel$score[1:14]) #normality for shared task: p-value<0.05 so not normal
shapiro.test(scores_permodel$score[1:28]) #normality for combined groups task: p-value<0.05 so not normal

#homoscedasticity not OK, normality not OK (not everywhere) 
#=> Mann-Whitney-U / Wilcoxon test on paired data
wilcox.test(score~type, data=scores_permodel[1:28,], paired=TRUE, var.equals=TRUE)
#p-value>0.05 so NO SIGNIFICANT DIFFERENCE BETWEEN OUR MODEL AND THE SHARED TASK



#Indo-European languages seem to do better, quantifying this with a one-way ANOVA
#checking assumptions of normality and homoscedasticity
leveneTest(fulldata$byt5 ~ fulldata$family) #p-value>0.05 so homoscedasticity OK
shapiro.test(fulldata$byt5) #p-value>0.05 so normality OK
model <- aov(byt5 ~ family, data=fulldata)
Anova(model)
#p-value>0.05 so NOT SIGNIFICANT but still a trend


################################################################################
###################         Effect of training data          ###################
################################################################################


#improvement achieved by using byte segmentation
fulldata$improvement_usingbytes = fulldata$byt5 - fulldata$sharedtask
#improvement achieved by including the features in the training set
fulldata$improvement_trainonfeatures = fulldata$byt5 - fulldata$nofeatures


#is there a significant improvement from including the features in the training?
leveneTest(score~type, data=scores_permodel[c(1:14, 29:42),]) #p-value>0.05 so homoscedastic
shapiro.test(scores_permodel$score[29:42]) #normality for ByT5 without features: p-value<0.05 so not normal
shapiro.test(scores_permodel$score[c(1:14, 29:42)]) #normality for combined group: p-value<0.05 so not normal

#homoscedasticity not OK, normality not OK => Mann-Whitney-U/Wilcoxon test on paired data
wilcox.test(score~type, data=scores_permodel[c(1:14, 29:42),], paired=TRUE, var.equals=TRUE)
#===> p-value<0.05 so SIGNIFICANT DIFFERENCE AFTER ADDING THE FEATURES



#training set size x byte accuracy, colouring by byte-related improvement
ggplot(fulldata, aes(x=trainingsetsize, y=byt5, colour=improvement_usingbytes))+ 
 geom_text(aes(label = iso_code), size=8)+  scale_color_gradientn(colors=c(low="black",high="magenta"))+
  xlab("Size of training set (billions of tokens)") + ylab("Accuracy")+
  theme(legend.text = element_text(size = 10)) + labs(color="Improvement from bytes")

#correlation between training set size and improvement from the bytes
cor.test(fulldata$trainingsetsize, fulldata$improvement_usingbytes, method="pearson")
#p-value>0.05 so NOT A SIGNIFICANT CORRELATION


#training set size x byte accuracy, colouring by feature-inclusion-related improvement
ggplot(fulldata, aes(x=trainingsetsize, y=byt5, colour=improvement_trainonfeatures))+ 
  geom_text(aes(label = iso_code), size=8)+  scale_color_gradientn(colors=c(low="black",high="green"))+
  xlab("Size of training set (billions of tokens)") + ylab("Accuracy")+
  theme(legend.text = element_text(size = 10)) + labs(color="Improvement from feature training")

#correlation between training set size and improvement from training on the features
cor.test(fulldata$trainingsetsize, fulldata$improvement_trainonfeat, method="pearson")
#p-value>0.05 so NOT A SIGNIFICANT CORRELATION

#is there a correlation between the improvement from the bytes and the improvement from the features?
cor.test(fulldata$improvement_trainonfeatures, fulldata$improvement_usingbytes, method="pearson")
#p-value>0.05 so NOT SIGNIFICANT

#checking if removing English (outlier) changes things
#we also performed correlations using Spearman's Rho, but didn't change the results
#so double checking by excluding the outlier altogether
woenglish=fulldata[c(1, 3:14),]

#correlation between training size and bytes improvement
cor.test(woenglish$trainingsetsize, woenglish$improvement_usingbytes, method="pearson")
#not a significant correlation, p-value>0.05

#correlation between training size and feature inclusion improvement
cor.test(woenglish$trainingsetsize, woenglish$improvement_trainonfeatures, method="pearson")
#not a significant correlation, p-value>0.05

#correlation between the improvements
cor.test(woenglish$improvement_usingbytes, woenglish$improvement_trainonfeatures, method="pearson")
#significant correlation, p-value<0.05

#===> removing English does not change the non-significance of the results
