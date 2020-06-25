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
            if i % opt.print_loss_freq_iter == opt.print_loss_freq_iter -1:
                losses = model.get_current_losses()
                model.print_current_losses(epoch,i,losses)
            if i % opt.save_cycleplot_freq_iter == opt.save_cycleplot_freq_iter -1:
                model.visual_iter(epoch,i)
        if epoch % opt.val_test_freq_epoch == opt.save_val_freq_epoch -1:
            with torch.no_grad():
                for k,test_data in enumerate(model.test_loader):
                    model.val_input(test_data)
                    model.forward()
                    model.visual_val(epoch,k)
        if epoch % opt.save_model_freq_epoch == opt.save_model_freq_epoch -1:
            model.save_models(epoch)
