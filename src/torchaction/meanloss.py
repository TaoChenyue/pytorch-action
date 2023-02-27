
class MeanLoss:
    def __init__(self,loss_fun) -> None:
        self.reset()
        self.loss_fun = loss_fun
        
    def __call__(self,*args):
        loss = self.loss_fun(*args)
        self.losses.append(loss.item())
        loss.backward()
        return loss.item()
    def reset(self):
        self.losses=[]        
    
    def compute(self):
        return sum(self.losses)/len(self.losses)
    
    
if __name__=='__main__':
    loss=MeanLoss(lambda x: x+1)
    print(loss(1))