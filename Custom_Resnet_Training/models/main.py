
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_transformations.data_transform_cifar10_custom_resnet import get_train_transform, get_test_transform
from data_loaders.cifar10_data_loader import get_train_loader, get_test_loader, get_classes



def train_test_loader(BATCH_SIZE=512):
    transform_train = get_train_transform()
    transform_test = get_test_transform()
    trainloader = get_train_loader(BATCH_SIZE, transform_train)
    testloader = get_test_loader(BATCH_SIZE, transform_test)
    classes = get_classes()

def get_model() :   
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  CustomResNet().to(device)
    return model

def perform_training(train_losses, test_losses,train_acc,test_acc,epochs=40, lr=0.01, momentum=0.9):
    model =  ResNet18().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

    EPOCHS = epochs
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, trainloader, optimizer, epoch, train_losses,train_acc )
        test(model, device, testloader, test_losses, test_acc)
        scheduler.step()
    return train_losses, test_losses,train_acc,test_acc

def incorrect_image(incorrect_image_list, predicted_label_list, correct_label_list):
        for (i, [data, target]) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze(1)         
            idxs_mask = (pred !=  target).view(-1)
            img_nm = data[idxs_mask].cpu().numpy()
            img_nm = img_nm.reshape(img_nm.shape[0], 3, 32, 32)
            if img_nm.shape[0] > 0:
                img_list = [img_nm[i] for i in range(img_nm.shape[0])]
                incorrect_image_list.extend(img_list)
                predicted_label_list.extend(pred[idxs_mask].detach().cpu().numpy())
                correct_label_list.extend(target[idxs_mask].detach().cpu().numpy())
        return incorrect_image_list, predicted_label_list, correct_label_list
        
def lr_range_test(PATH_BASE_MODEL,model,EPOCHS_TO_TRY,max_lr_list, test_accuracy_list):
    for lr_value in max_lr_list:
        model.load_state_dict(torch.load(PATH_BASE_MODEL))
        optimizer = optim.SGD(model.parameters(), lr=lr_value/10, momentum=0.9)

        lr_finder_schedule = lambda t: np.interp([t], [0, EPOCHS_TO_TRY], [lr_value/10,  lr_value])[0]
        lr_finder_lambda = lambda it: lr_finder_schedule(it * BATCH_SIZE/50176)
        max_lr_finder = max_lr_finder_schedule(no_of_images=50176, batch_size=BATCH_SIZE, base_lr=lr_value/10, max_lr=lr_value, total_epochs=5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[max_lr_finder])
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for epoch in range(EPOCHS_TO_TRY):
            print("MAX LR:" ,lr_value, " EPOCH:", (epoch+1))        
            train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc,True )
            test(model, device, testloader, test_losses, test_acc)
        t_acc = test_acc[-1]
        test_accuracy_list.append(t_acc)
        print(" For Max LR: ", lr_value, " Test Accuracy: ", t_acc)
    return max_lr_list, test_accuracy_list,model
    
def train(model,optimizer,EPOCHS, one_cyle_lr,scheduler,PATH,best_test_accuracy,train_losses,test_losses,train_acc,test_acc):
    for epoch in range(EPOCHS):
        print("EPOCH:", (epoch+1))
        train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc, True )
        test(model, device, testloader, test_losses, test_acc)
        t_acc = test_acc[-1]
        if t_acc > best_test_accuracy:
            print("Test Accuracy: " + str(t_acc) + " has increased. Saving the model")
            best_test_accuracy = t_acc
            torch.save(model.state_dict(), PATH)
    return train_losses,test_losses,train_acc,test_acc
