# rscript libs/ToolboxModules/callbfast.R height width startyear endyear frequency tile feature batchsize outpath bfastpath

# read arguments
args <- commandArgs(trailingOnly=TRUE)

# test if there is correct number of arguments: if not, return an error
if (length(args)!=10){
	stop("Wrong number of arguments", call.=FALSE)
}

height <- as.numeric(args[1])
width <- as.numeric(args[2])
startyear <- as.numeric(args[3])
endyear <- as.numeric(args[4])
freq <- as.numeric(args[5])
tile <- args[6]
feature <- args[7]
batchsize <- as.numeric(args[8])
outpath <- args[9]
bfastpath <- args[10]

npixels <- height*width

# parallel processing setup
if(!require(doParallel)){
	install.packages("doParallel")
	library(doParallel)
}
cores <- detectCores()
cl <- makeCluster(cores[1]/2 - 1)
registerDoParallel(cl)

# combine results function
comb <- function(x, ...) {
  lapply(seq_along(x),
    function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}

if(!require(rhdf5)){
	install.packages("rhdf5")
	library(rhdf5)
}

changeband <- matrix(, nrow=height, ncol=width)
accuracyband <- matrix(, nrow=height, ncol=width)

for(b in seq(1, npixels, by=batchsize)){
	# read .h5 files and load batch data
	data <- matrix()    # initialize empty matrix

	for(y in startyear:endyear){
		tilename <- paste(tile, as.character(y), sep="_", collapse = NULL)
		featurename <- paste('NDI', feature, sep="", collapse = NULL)
		loadpath <- paste(outpath, tilename, 'NDI_TimeSeries', featurename, 'ts.h5', sep = "/", collapse = NULL)

		h5f = H5Fopen(loadpath)
		h5d = h5f&"/ts"    # return dataset handle for matrix ts

		if(all(is.na(data))){
			data <- t(h5d[,b:(b+batchsize-1)])   # ts matrix in .h5 is transposed
		} else{
			yeardata <- t(h5d[,b:(b+batchsize-1)])   # ts matrix in .h5 is transposed
			yeardata <- yeardata[,-c(1,2,3)]
			data <- cbind(data, yeardata)
		}
		h5closeAll()
	}

	# parallel change detection for pixels in batch
	results <- foreach(npx=1:batchsize, .combine='comb', .multicombine=TRUE,
                .init=list(list(), list())) %dopar% {
		# get pixel time series
		datapx <- data[npx,]
		datapx <- datapx[-c(1,2,3)]

		# check for NaN values
		if(any(is.nan(datapx))){
			change <- 0
			accuracy <- 0
		} else{
			source(paste(bfastpath, '/R/bfast.R', sep="/", collapse = NULL))
			source(paste(bfastpath, '/R/print.bfast.R', sep="/", collapse = NULL))

			if(!require(lubridate, warn.conflicts = FALSE)){
				install.packages("lubridate")
				library(lubridate, warn.conflicts = FALSE)
			}
			if(!require(zoo, warn.conflicts = FALSE)){
				install.packages("zoo")
				library(zoo, warn.conflicts = FALSE)
			}
			if(!require(sandwich)){
				install.packages("sandwich")
				library(sandwich)
			}
			if(!require(strucchange)){
				install.packages("strucchange")
				library(strucchange)
			}
			if(!require(xts, warn.conflicts = FALSE)){
				install.packages("xts")
				library(xts, warn.conflicts = FALSE)
			}

			# farming year: start 11 november (315), end 10 november (314)
			date.string <- paste(as.character(startyear-1), "-11-11", sep='')
			tsdata <- ts(datapx, frequency=365, start=decimal_date(ymd(date.string)))

			# daily to weekly aggregation
			if(freq==52){
				ts.daily <- tsdata
				dates <- as.Date(date_decimal(as.numeric(time(ts.daily))))
				xts.daily <- xts(ts.daily, order.by = dates)
				xts.weekly <- xts::apply.weekly(xts.daily, median)  # retrieve median value for each week
				ts.weekly <- ts(as.numeric(xts.weekly), 
								# define the start and end (Year, Week)    
								start = c(as.numeric(format(start(xts.weekly), "%Y")), 
										as.numeric(format(start(xts.weekly), "%W"))), 
								end   = c(as.numeric(format(end(xts.weekly), "%Y")), 
										as.numeric(format(end(xts.weekly), "%W"))), 
								frequency = 52)
				tsdata <- ts.weekly
			}

			rdist <- (freq/2)/length(tsdata)
			err <- try({
				fit <- bfast(tsdata, h=rdist, season="none", max.iter=1, breaks=1)
			}, silent=TRUE)

			if(class(err) == 'try-error'){
				change <- 0
				accuracy <- 0
			}
			else{
				if(fit$output[[1]]$Vt.bp[1] != 0){
					#breakpoint
					brk <- fit$output[[1]]$Vt.bp[1]

					# change
					change <- ceiling(brk/freq)

					# accuracy
					lv <- 0.99
					ci <- confint(object = fit$output[[1]]$bp.Vt, level = lv, het.err = FALSE)
					while((ci$confint[3] - ci$confint[1]) > 4){
					    lv = lv - 0.01
					    ci <- confint(object = fit$output[[1]]$bp.Vt, level = lv, het.err = FALSE)
					}
					accuracy <- lv*100
				} else{
					change <- 0
					accuracy <- 0
				}
			}
		}
		list(change, accuracy)
	}

	changelist <- results[[1]]
	accuracylist <- results[[2]]
	
	for(npx in 1:batchsize){
		id <- data[npx,1]
		row <- id %/% width
		col <- id %% width
		changeband[row+1,col+1] <- changelist[[npx]]
		accuracyband[row+1,col+1] <- accuracylist[[npx]]
	}
}

stopCluster(cl)

# save output image
if(!require(raster)){
	install.packages(raster)
	library(raster)
}
r1 <- raster(changeband)
r2 <- raster(accuracyband)
s <- stack(r1,r2)

featurename <- paste('NDI', feature, sep="", collapse = NULL)
filename <- paste(tile, 'CD', featurename, sep = "_", collapse = NULL)
dir.create(file.path(outpath, tile), showWarnings = FALSE)
savepath <- paste(outpath, tile, filename, sep = "/", collapse = NULL)

writeRaster(s, savepath, format = "GTiff")