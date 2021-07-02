
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



def train_test_loader():
    transform_train = get_train_transform()
    transform_test = get_test_transform()
    trainloader = get_train_loader(256, transform_train)
    testloader = get_test_loader(256, transform_test)
    classes = get_classes()

def get_model_parameters() :   
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18().to(device)
    return model

def perform_training(epochs=40, lr=0.01, momentum=0.9,train_losses, test_losses,train_acc,test_acc):
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