from wilds.common.data_loaders import get_eval_loader
from wilds.common.data_loaders import get_train_loader
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
import torchvision.transforms as transforms
import torch

def get_wilds_dataloader(data_name, args):
	"""

	Args:
		data_name ([type]): {"iwildcam"}
		args ([type]): [description]

	Returns:
		[type]: [description]
	"""
	dataset = get_dataset(dataset=data_name, download=True)

	# Get the training set
	train_data = dataset.get_subset(
		"train",
		transform=transforms.Compose(
			[transforms.Resize((32, 32)), transforms.ToTensor()]
		),
	)

	test_data = dataset.get_subset(
		"test",
		transform=transforms.Compose(
			[transforms.Resize((32, 32)), transforms.ToTensor()]
		),
	)
	print(f"Training N: {len(train_data)}, test N: {len(test_data)}")
	# Prepare the standard data loader
	train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
	test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
	return train_loader, test_loader


def get_cifar10_dataloader(args):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

	testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
	return trainloader, testloader


class CIFARC_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        if transform != None:
            self.images = []
            for img in images:
                self.images.append(transform(img))
            self.images = torch.stack(self.images)
        else:
            self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, key):
        if type(key) == slice:
            return CIFARC_Dataset(self.images[key], self.labels[key])


def get_cifarC_dataloader(data_name='cifar10'):
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	dataset_name = 'CIFAR-100-C' if data_name.lower() == 'cifar100' else 'CIFAR-10-C'
	#* CIFAR-C Data loaders
	print('==> Preparing data..')
	shift_files = glob.glob(f'data/{dataset_name}/*.npy')
	shift_files.remove(f'data/{dataset_name}/labels.npy')
	label = torch.from_numpy(np.load(f'data/{dataset_name}/labels.npy'))
	loaders = {'1':[], '2':[], '3':[], '4':[], '5':[]}

	for file in shift_files:
		print(f"Make loaders for {file}")
		image = np.load(file)#).transpose(3,1).transpose(3,2) # Make NCHW
		testset = CIFAR100C_Dataset(image, label, transform=transform_test)
		for i in range(5):
			print(f"Shift intensity : [{i+1}]")
			testloader = torch.utils.data.DataLoader(testset[i*10000: (i+1)*10000], batch_size=100
				, shuffle=False)
			loaders[f'{i+1}'].append(testloader)
	return loaders