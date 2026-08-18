standardize_ws = function(w,w_tilde){
        n = nrow(w)
        n_tilde = nrow(w_tilde)

        mu_w = colMeans(w)
        sigma_w = sqrt(colMeans(w^2)-mu_w^2)

        w = (w - matrix(mu_w,nrow=n,ncol=length(mu_w),byrow=TRUE))/matrix(sigma_w,nrow=n,ncol=length(mu_w),byrow=TRUE)
        w_tilde = (w_tilde - matrix(mu_w,nrow=n_tilde,ncol=length(mu_w),byrow=TRUE))/matrix(sigma_w,nrow=n_tilde,ncol=length(mu_w),byrow=TRUE)

        return(list(w=w,w_tilde=w_tilde))
}

gen_data_owl = function(n,setting,wdim=10,n_tilde=1000,...){
  n_all = n + n_tilde
  train_inds = 1:n
  test_inds = (n+1):n_all

  if(wdim<8){
        stop("gen_data_owl requires wdim>=8.")
  }

  # simulate data
  w_all = matrix(runif(n_all*wdim,min=-1,max=1),nrow=n_all)
  a_all = rbinom(n_all,1,1/2)
  if(setting==1){
    mm = 0.442 * (1-w_all[,1] - w_all[,2])
  } else if (setting==2) {
    mm = w_all[,2] - 0.25*w_all[,1]^2 - 1
  } else if (setting==3) {
    mm = (0.5 - w_all[,1]^2 - w_all[,2]^2) * (w_all[,1]^2 + w_all[,2]^2 - 0.3)
  } else if (setting==4) {
    mm = 1 - w_all[,1]^3 + exp(w_all[,3]^2 + w_all[,5]) + 0.6*w_all[,6] - (w_all[,7] + w_all[,8])^2
  }
  y_all = mm * (2*a_all-1) + rnorm(n_all)
  cate_all = 2*mm

  return(c(standardize_ws(w_all[train_inds,],w_all[test_inds,]),list(a=a_all[train_inds],y=y_all[train_inds],cate_tilde = cate_all[test_inds])))
}

out = gen_data_owl(100,setting=1)

if(FALSE){
  library(ggplot2)
  library(interp)
  df = data.frame(
    w1 = out$w_tilde[,1],
    w2 = out$w_tilde[,3],
    cate = out$cate_tilde)
  
  # ggplot(df,aes(w1,w2,z = cate)) + theme_bw() + geom_contour_filled()
  
  grid <- with(df, interp::interp(w1, w2, cate))
  
  griddf <- subset(data.frame(x = rep(grid$x, nrow(grid$z)),
                              y = rep(grid$y, each = ncol(grid$z)),
                              z = as.numeric(grid$z)),
                   !is.na(z))
  ggplot(griddf,aes(x, y, z=z)) + theme_bw() + geom_contour_filled()
}

library(grf)
library(flam)
library(parallel)

num_mc = 1000

df = NULL
mses_list = list()
for(setting in 1:4){
  mses = do.call(cbind,mclapply(1:num_mc,function(i){
    print(c(setting,i))
    out = gen_data_owl(500,setting=setting)
  
    # Linear regression
    df = with(out,data.frame(w,a,y))
    lm_fit0 = lm(y~.,data=subset(df,a==0,select=-c(a)))
    lm_fit1 = lm(y~.,data=subset(df,a==1,select=-c(a)))
    lm_mse = mean((out$cate_tilde - (predict(lm_fit1,newdata=data.frame(out$w_tilde)) - predict(lm_fit0,newdata=data.frame(out$w_tilde))))^2)
  
    # FLAM
    invisible(capture.output(flam_fit0 <- with(out,flamCV(w[a==0,],y[a==0]))))
    invisible(capture.output(flam_fit1 <- with(out,flamCV(w[a==1,],y[a==1]))))
    flam_mse = mean((out$cate_tilde - (with(flam_fit1,predict(flam.out,new.x=out$w_tilde,lambda=lambda.cv,alpha=alpha)) - with(flam_fit0,predict(flam.out,new.x=out$w_tilde,lambda=lambda.cv,alpha=alpha))))^2)
    
    # Causal forest
    tau.forest <- causal_forest(out$w, out$y, out$a,tune.parameters='all')
    cf_mse = mean((out$cate_tilde-predict(tau.forest,newdata=out$w_tilde)$predictions)^2)
    
    return(c('lm'=lm_mse,'flam'=flam_mse,'cf'=cf_mse))
  },mc.cores=detectCores()-4))
  
  mses_list = append(mses_list,mses)
    
  mse = rowMeans(mses)
  se = apply(mses,1,sd)/sqrt(num_mc)
  df_curr = data.frame('setting'=setting,'method'=names(mse),'mse'=mse,'lb'=mse-qnorm(0.975)*se,'ub'=mse+qnorm(0.975)*se)
  rownames(df_curr) = NULL
  df = rbind(df,df_curr)
  
  print(df)
}

save(df,mses_list,file="owl_results.Rdata")
