mynet = coder.loadDeepLearningNetwork('alexnet_model.mat', 'alexnet');

imds = imageDatastore('test');
inputSize = mynet.Layers(1).InputSize;

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imds);

[YPred,scores] = classify(mynet,augimdsValidation);

idx = randperm(numel(augimdsValidation.Files),4);

figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imds,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end