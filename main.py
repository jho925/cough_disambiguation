from model import CoughClassifier, training, test, get_dataloader
import torch

def main():
	Model = CoughClassifier()
	train_dl, test_dl = get_dataloader('flusense_train_speech.csv','flusense_test.csv')

	training(Model, train_dl, 30)
	torch.save(Model.state_dict(), 'model.pth')
	test(Model, test_dl)

if __name__ == '__main__':
	main()