import torch
from models.options import ParamOptions
from models.trainer import TrainModel

if __name__ == '__main__':
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    model.init_model()
    for epoch in range(opt.num_epochs):
        model.update_learning_rate(epoch)
        for i, train_data in enumerate(model.train_loader):
            model.set_input(train_data)
            model.optimization()

            losses = model.get_current_losses()
            model.print_current_losses(epoch,i,losses)
            if i%100 ==99:
                model.visual_iter(epoch,i)
        with torch.no_grad():
            test_count = 0
            for i,test_data in enumerate(model.test_loader):
                model.set_input(test_data)
                model.forward()
                model.visual_iter(epoch,i)
