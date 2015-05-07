# Script to calculate the number of frames for a folder structure containing images
# in single frame png format

root_dir <- "/Volumes/Data/opticalflow/CK/cohn-kanade"

files <- list.files(path=root_dir)

nestedFolders <- lapply(files, function(folder){ paste(root_dir, folder, list.files(path=paste(root_dir, folder, sep="/")), sep="/")})

allFolders <- unlist(nestedFolders, recursive=FALSE)

allPngs <- mapply(function(folder){list.files(path=folder,pattern="png")}, allFolders)

numberOfFrames <- mapply(function(l){length(l)}, allPngs)

hist(numberOfFrames, main="Distribution of the number of frames over videos", xlab="Number of Frames")

summary(numberOfFrames)