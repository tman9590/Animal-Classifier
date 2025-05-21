import os
import json
import torch
from torch import nn
import urllib.request
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# model and network found in this repo
# https://github.com/Moddy2024/Bird-Classification/blob/main/prediction.ipynb

bird_name_map = {
    0: 'Abbotts Babbler', 1: 'Abbotts Booby', 2: 'Abyssinian Ground Hornbill', 3: 'African Crowned Crane',
    4: 'African Emerald Cuckoo', 5: 'African Firefinch', 6: 'African Oyster Catcher', 7: 'African Pied Hornbill',
    8: 'Albatross', 9: 'Alberts Towhee', 10: 'Alexandrine Parakeet', 11: 'Alpine Chough',
    12: 'Altamira Yellowthroat', 13: 'American Avocet', 14: 'American Bittern', 15: 'American Coot',
    16: 'American Flamingo', 17: 'American Goldfinch', 18: 'American Kestrel', 19: 'American Pipit',
    20: 'American Redstart', 21: 'American Wigeon', 22: 'Amethyst Woodstar', 23: 'Andean Goose',
    24: 'Andean Lapwing', 25: 'Andean Siskin', 26: 'Anhinga', 27: 'Anianiau', 28: 'Annas Hummingbird',
    29: 'Antbird', 30: 'Antillean Euphonia', 31: 'Apapane', 32: 'Apostlebird', 33: 'Araripe Manakin',
    34: 'Ashy Storm Petrel', 35: 'Ashy Thrushbird', 36: 'Asian Crested Ibis', 37: 'Asian Dollard Bird',
    38: 'Auckland Shaq', 39: 'Austral Canastero', 40: 'Australasian Figbird', 41: 'Avadavat',
    42: 'Azaras Spinetail', 43: 'Azure Breasted Pitta', 44: 'Azure Jay', 45: 'Azure Tanager',
    46: 'Azure Tit', 47: 'Baikal Teal', 48: 'Bald Eagle', 49: 'Bald Ibis', 50: 'Bali Starling',
    51: 'Baltimore Oriole', 52: 'Bananaquit', 53: 'Band Tailed Guan', 54: 'Banded Broadbill',
    55: 'Banded Pita', 56: 'Banded Stilt', 57: 'Bar-tailed Godwit', 58: 'Barn Owl', 59: 'Barn Swallow',
    60: 'Barred Puffbird', 61: 'Barrows Goldeneye', 62: 'Bay-breasted Warbler', 63: 'Bearded Barbet',
    64: 'Bearded Bellbird', 65: 'Bearded Reedling', 66: 'Belted Kingfisher', 67: 'Bird of Paradise',
    68: 'Black & Yellow Broadbill', 69: 'Black Baza', 70: 'Black Cockato', 71: 'Black Francolin',
    72: 'Black Skimmer', 73: 'Black Swan', 74: 'Black Tail Crake', 75: 'Black Throated Bushtit',
    76: 'Black Throated Warbler', 77: 'Black Vented Shearwater', 78: 'Black Vulture',
    79: 'Black-capped Chickadee', 80: 'Black-necked Grebe', 81: 'Black-throated Sparrow',
    82: 'Blackburniam Warbler', 83: 'Blonde Crested Woodpecker', 84: 'Blood Pheasant', 85: 'Blue Coau',
    86: 'Blue Dacnis', 87: 'Blue Grouse', 88: 'Blue Heron', 89: 'Blue Malkoha', 90: 'Blue Throated Toucanet',
    91: 'Bobolink', 92: 'Bornean Bristlehead', 93: 'Bornean Leafbird', 94: 'Bornean Pheasant',
    95: 'Brandt Cormarant', 96: 'Brewers Blackbird', 97: 'Brown Crepper', 98: 'Brown Noody',
    99: 'Brown Thrasher', 100: 'Bufflehead', 101: 'Bulwers Pheasant', 102: 'Burchells Courser',
    103: 'Bush Turkey', 104: 'Caatinga Cacholote', 105: 'Cactus Wren', 106: 'California Condor',
    107: 'California Gull', 108: 'California Quail', 109: 'Campo Flicker', 110: 'Canary',
    111: 'Cape Glossy Starling', 112: 'Cape Longclaw', 113: 'Cape May Warbler', 114: 'Cape Rock Thrush',
    115: 'Capped Heron', 116: 'Capuchinbird', 117: 'Carmine Bee-eater', 118: 'Caspian Tern',
    119: 'Cassowary', 120: 'Cedar Waxwing', 121: 'Cerulean Warbler', 122: 'Chara de Collar',
    123: 'Chattering Lory', 124: 'Chestnet Bellied Euphonia', 125: 'Chinese Bamboo Partridge',
    126: 'Chinese Pond Heron', 127: 'Chipping Sparrow', 128: 'Chucao Tapaculo', 129: 'Chukar Partridge',
    130: 'Cinnamon Attila', 131: 'Cinnamon Flycatcher', 132: 'Cinnamon Teal', 133: 'Clarks Nutcracker',
    134: 'Cock of the Rock', 135: 'Cockatoo', 136: 'Collared Aracari', 137: 'Common Firecrest',
    138: 'Common Grackle', 139: 'Common House Martin', 140: 'Common Iora', 141: 'Common Loon',
    142: 'Common Poorwill', 143: 'Common Starling', 144: 'Coppery Tailed Coucal', 145: 'Crab Plover',
    146: 'Crane Hawk', 147: 'Cream Colored Woodpecker', 148: 'Crested Auklet', 149: 'Crested Caracara',
    150: 'Crested Coua', 151: 'Crested Fireback', 152: 'Crested Kingfisher', 153: 'Crested Nuthatch',
    154: 'Crested Oropendola', 155: 'Crested Shriketit', 156: 'Crimson Chat', 157: 'Crimson Sunbird',
    158: 'Crow', 159: 'Crowned Pigeon', 160: 'Cuban Tody', 161: 'Cuban Trogon', 162: 'Curl Crested Aracuri',
    163: "D-Arnauds Barbet", 164: 'Dalmatian Pelican', 165: 'Darjeeling Woodpecker',
    166: 'Dark Eyed Junco', 167: 'Darwins Flycatcher', 168: 'Daurian Redstart', 169: 'Demoiselle Crane',
    170: 'Double Barred Finch', 171: 'Double Brested Cormarant', 172: 'Double Eyed Fig Parrot',
    173: 'Downy Woodpecker', 174: 'Dusky Lory', 175: 'Dusky Robin', 176: 'Eared Pita',
    177: 'Eastern Bluebird', 178: 'Eastern Bluebonnet', 179: 'Eastern Golden Weaver',
    180: 'Eastern Meadowlark', 181: 'Eastern Rosella', 182: 'Eastern Towee', 183: 'Eastern Wip Poor Will',
    184: 'Ecuadorian Hillstar', 185: 'Egyptian Goose', 186: 'Elegant Trogon', 187: 'Elliots Pheasant',
    188: 'Emerald Tanager', 189: 'Emperor Penguin', 190: 'Emu', 191: 'Enggano Myna',
    192: 'Eurasian Bullfinch', 193: 'Eurasian Golden Oriole', 194: 'Eurasian Magpie',
    195: 'European Goldfinch', 196: 'European Turtle Dove', 197: 'Evening Grosbeak',
    198: 'Fairy Bluebird', 199: 'Fairy Penguin', 200: 'Fairy Tern', 201: 'Fan Tailed Widow',
    202: 'Fasciated Wren', 203: 'Fiery Minivet', 204: 'Fiordland Penguin', 205: 'Fire Tailled Myzornis',
    206: 'Flame Bowerbird', 207: 'Flame Tanager', 208: 'Frigate', 209: 'Gambels Quail',
    210: 'Gang Gang Cockatoo', 211: 'Gila Woodpecker', 212: 'Gilded Flicker', 213: 'Glossy Ibis',
    214: 'Go Away Bird', 215: 'Gold Wing Warbler', 216: 'Golden Bower Bird', 217: 'Golden Cheeked Warbler',
    218: 'Golden Chlorophonia', 219: 'Golden Eagle', 220: 'Golden Parakeet', 221: 'Golden Pheasant',
    222: 'Golden Pipit', 223: 'Gouldian Finch', 224: 'Grandala', 225: 'Gray Catbird', 226: 'Gray Kingbird',
    227: 'Gray Partridge', 228: 'Great Gray Owl', 229: 'Great Jacamar', 230: 'Great Kiskadee',
    231: 'Great Potoo', 232: 'Great Tinamou', 233: 'Great Xenops', 234: 'Greater Pewee',
    235: 'Greator Sage Grouse', 236: 'Green Broadbill', 237: 'Green Jay', 238: 'Green Magpie',
    239: 'Grey Cuckooshrike', 240: 'Grey Plover', 241: 'Groved Billed Ani', 242: 'Guinea Turaco',
    243: 'Guineafowl', 244: 'Gurneys Pitta', 245: 'Gyrfalcon', 246: 'Hamerkop', 247: 'Harlequin Duck',
    248: 'Harlequin Quail', 249: 'Harpy Eagle', 250: 'Hawaiian Goose', 251: 'Hawfinch',
    252: 'Helmet Vanga', 253: 'Hepatic Tanager', 254: 'Himalayan Bluetail', 255: 'Himalayan Monal',
    256: 'Hoatzin', 257: 'Hooded Merganser', 258: 'Hoopoes', 259: 'Horned Guan', 260: 'Horned Lark',
    261: 'Horned Sungem', 262: 'House Finch', 263: 'House Sparrow', 264: 'Hyacinth Macaw',
    265: 'Iberian Magpie', 266: 'Ibisbill', 267: 'Imperial Shaq', 268: 'Inca Tern', 269: 'Indian Bustard',
    270: 'Indian Pitta', 271: 'Indian Roller', 272: 'Indian Vulture', 273: 'Indigo Bunting',
    274: 'Indigo Flycatcher', 275: 'Inland Dotterel', 276: 'Ivory Billed Aracari', 277: 'Ivory Gull',
    278: 'Iwi', 279: 'Jabiru', 280: 'Jack Snipe', 281: 'Jandaya Parakeet', 282: 'Japanese Robin',
    283: 'Java Sparrow', 284: 'Jocotoco Antpitta', 285: 'Kagu', 286: 'Kakapo', 287: 'Killdear',
    288: 'King Eider', 289: 'King Vulture', 290: 'Kiwi', 291: 'Kookaburra', 292: 'Lark Bunting',
    293: 'Lazuli Bunting', 294: 'Lesser Adjutant', 295: 'Lilac Roller', 296: 'Little Auk',
    297: 'Loggerhead Shrike', 298: 'Long-eared Owl', 299: 'Magpie Goose', 300: 'Malabar Hornbill',
    301: 'Malachite Kingfisher', 302: 'Malagasy White Eye', 303: 'Maleo', 304: 'Mallard Duck',
    305: 'Mandrin Duck', 306: 'Mangrove Cuckoo', 307: 'Marabou Stork', 308: 'Masked Booby',
    309: 'Masked Lapwing', 310: 'Mckays Bunting', 311: 'Mikado Pheasant', 312: 'Mourning Dove',
    313: 'Myna', 314: 'Nicobar Pigeon', 315: 'Noisy Friarbird', 316: 'Northern Beardless Tyrannulet',
    317: 'Northern Cardinal', 318: 'Northern Flicker', 319: 'Northern Fulmar', 320: 'Northern Gannet',
    321: 'Northern Goshawk', 322: 'Northern Jacana', 323: 'Northern Mockingbird', 324: 'Northern Parula',
    325: 'Northern Red Bishop', 326: 'Northern Shoveler', 327: 'Ocellated Turkey', 328: 'Okinawa Rail',
    329: 'Orange Brested Bunting', 330: 'Oriental Bay Owl', 331: 'Osprey', 332: 'Ostrich',
    333: 'Ovenbird', 334: 'Oyster Catcher', 335: 'Painted Bunting', 336: 'Palila', 337: 'Paradise Tanager',
    338: 'Parakett Akulet', 339: 'Parus Major', 340: 'Patagonian Sierra Finch', 341: 'Peacock',
    342: 'Peregrine Falcon', 343: 'Philippine Eagle', 344: 'Pink Robin', 345: 'Pomarine Jaeger',
    346: 'Puffin', 347: 'Purple Finch', 348: 'Purple Gallinule', 349: 'Purple Martin',
    350: 'Purple Swamphen', 351: 'Pygmy Kingfisher', 352: 'Quetzal', 353: 'Rainbow Lorikeet',
    354: 'Razorbill', 355: 'Red Bearded Bee Eater', 356: 'Red Bellied Pitta', 357: 'Red Browed Finch',
    358: 'Red Faced Cormorant', 359: 'Red Faced Warbler', 360: 'Red Fody', 361: 'Red Headed Duck',
    362: 'Red Headed Woodpecker', 363: 'Red Honey Creeper', 364: 'Red Naped Trogon',
    365: 'Red Tailed Hawk', 366: 'Red Tailed Thrush', 367: 'Red Winged Blackbird',
    368: 'Red Wiskered Bulbul', 369: 'Regent Bowerbird', 370: 'Ring-necked Pheasant',
    371: 'Roadrunner', 372: 'Robin', 373: 'Rock Dove', 374: 'Rosy Faced Lovebird',
    375: 'Rough Leg Buzzard', 376: 'Royal Flycatcher', 377: 'Ruby Throated Hummingbird',
    378: 'Rudy Kingfisher', 379: 'Rufous Kingfisher', 380: 'Rufuos Motmot', 381: 'Samatran Thrush',
    382: 'Sand Martin', 383: 'Sandhill Crane', 384: 'Satyr Tragopan', 385: 'Scarlet Crowned Fruit Dove',
    386: 'Scarlet Ibis', 387: 'Scarlet Macaw', 388: 'Scarlet Tanager', 389: 'Shoebill',
    390: 'Short Billed Dowitcher', 391: 'Skua', 392: 'Smiths Longspur', 393: 'Snowy Egret',
    394: 'Snowy Owl', 395: 'Snowy Plover', 396: 'Sora', 397: 'Spangled Cotinga', 398: 'Splendid Wren',
    399: 'Spoon Biled Sandpiper', 400: 'Spoonbill', 401: 'Spotted Catbird', 402: 'Sri Lanka Blue Magpie',
    403: 'Steamer Duck', 404: 'Stork Billed Kingfisher', 405: 'Strawberry Finch', 406: 'Striped Owl',
    407: 'Stripped Manakin', 408: 'Stripped Swallow', 409: 'Superb Starling', 410: 'Swinhoes Pheasant',
    411: 'Tailorbird', 412: 'Taiwan Magpie', 413: 'Takahe', 414: 'Tasmanian Hen', 415: 'Teal Duck',
    416: 'Tit Mouse', 417: 'Toucan', 418: 'Townsends Warbler', 419: 'Tree Swallow',
    420: 'Tricolored Blackbird', 421: 'Tropical Kingbird', 422: 'Trumpter Swan', 423: 'Turkey Vulture',
    424: 'Turquoise Motmot', 425: 'Umbrella Bird', 426: 'Varied Thrush', 427: 'Veery',
    428: 'Venezuelian Troupial', 429: 'Vermilion Flycather', 430: 'Victoria Crowned Pigeon',
    431: 'Violet Green Swallow', 432: 'Violet Turaco', 433: 'Vulturine Guineafowl',
    434: 'Wall Creaper', 435: 'Wattled Curassow', 436: 'Wattled Lapwing', 437: 'Whimbrel',
    438: 'White Browed Crake', 439: 'White Cheeked Turaco', 440: 'White Crested Hornbill',
    441: 'White Necked Raven', 442: 'White Tailed Tropic', 443: 'White Throated Bee Eater',
    444: 'Wild Turkey', 445: 'Wilsons Bird of Paradise', 446: 'Wood Duck',
    447: 'Yellow Bellied Flowerpecker', 448: 'Yellow Cacique', 449: 'Yellow Headed Blackbird'
}

def conv_block(in_channels, out_channels, activation=False, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels)]
    if activation: layers.append(nn.ReLU(inplace=True))
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet34(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=4),
            nn.BatchNorm2d(64),nn.MaxPool2d(2), nn.ReLU(inplace=True))
           
        self.res1 = nn.Sequential(conv_block(64, 64,activation=True), conv_block(64, 64))
        self.res2 = nn.Sequential(conv_block(64, 64,activation=True), conv_block(64, 64))
        self.res3 = nn.Sequential(conv_block(64, 64,activation=True), conv_block(64, 64))
        self.downsample1=nn.Sequential(conv_block(64, 128,pool=True)) 
        self.res4 = nn.Sequential(conv_block(64, 128,activation=True, pool=True),
                                  conv_block(128,128))
        self.res5 = nn.Sequential(conv_block(128, 128,activation=True), conv_block(128, 128))
        self.res6 = nn.Sequential(conv_block(128, 128,activation=True), conv_block(128, 128))
        self.res7 = nn.Sequential(conv_block(128, 128,activation=True), conv_block(128, 128))
        self.res8 = nn.Sequential(conv_block(128, 256,activation=True, pool=True),
                                  conv_block(256,256))
        self.downsample2 = nn.Sequential(conv_block(128, 256,pool=True))
        self.res9 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256))
        self.res10 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256))
        self.res11 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256))
        self.res12 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256))
        self.res13 = nn.Sequential(conv_block(256, 256,activation=True), conv_block(256, 256))
        self.res14 = nn.Sequential(conv_block(256, 512,activation=True, pool=True),
                                   conv_block(512,512))
        
        self.downsample3 = nn.Sequential(conv_block(256, 512,pool=True))
        self.res15 = nn.Sequential(conv_block(512, 512,activation=True), conv_block(512, 512))
        self.res16 = nn.Sequential(conv_block(512, 512,activation=True), conv_block(512, 512,activation=True))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.17),
                                        nn.Linear(512, num_classes))
      
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.res2(out) + out
        out = self.res3(out) + out
        out = self.downsample1(out) +self.res4(out)
        out = self.res5(out) + out
        out = self.res6(out) + out
        out = self.res7(out) + out
        out = self.downsample2(out) +self.res8(out)
        out = self.res9(out) + out
        out = self.res10(out) + out
        out = self.res11(out) + out
        out = self.res12(out) + out
        out = self.res13(out) + out
        out = self.downsample3(out) + self.res14(out) 
        out = self.res15(out) + out
        out = self.res16(out) + out
        out = self.classifier(out)
        return (out)
     

class BirdResnet(nn.Module):
    def __init__(self):
        super().__init__()
        # Using the pretrained model
        self.network = ResNet34(3,450)
    
    def forward(self, xb):
        return (self.network(xb))
     

model = BirdResnet()
model.load_state_dict(torch.load("bird-resnet34best.pth"))
model.eval()

input_shape = (1, 3, 224, 224)
img = Image.open('american-goldfinch.png')
img = img.convert("RGB")
img = img.resize((input_shape[2], input_shape[3]))
stats = ((0.4758, 0.4685, 0.3870), (0.2376, 0.2282, 0.2475))
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(*stats,inplace=True)]) 
img=transform(img)
img = img.unsqueeze(0)
example_input = img
traced_model = torch.jit.trace(model, example_input)
traced_model.eval();
output_data = traced_model(example_input)
o = output_data.softmax(dim=1)
result = torch.max(o, dim=1)
label = bird_name_map[result.indices.item()]
print(f"Predicted label: {label}")

model_config = {
    "input_shape": input_shape,
    "model": "resnet",
    "mean": stats[0],
    "std": stats[1],
    "files": [
    ],
    "labels": bird_name_map,
}

def export_openvino():
    import openvino as ov

    ov_model = ov.convert_model(traced_model, example_input=example_input, input=[input_shape])

    path = "models/openvino"
    os.system(f"rm -rf {path}")

    ov.save_model(ov_model, f"{path}/model.xml")
    model_config["files"] = [
        f"model.xml",
        f"model.bin"
    ]

    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_coreml():
    import coremltools as ct

    path = "models/coreml"
    os.system(f"rm -rf {path}")
    
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape)],
    )

    model.save(path + "/model.mlpackage")

    model_config["files"] = [
        f"model.mlpackage/Manifest.json",
        f"model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        f"model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_onnx():
    path = "models/onnx"

    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")

    torch.onnx.export(
        traced_model,
        example_input,
        f"{path}/model.onnx",
        verbose=False,
        input_names=["input"],
    )

    model_config["files"] = [
        f"model.onnx",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_ncnn():
    path = "models/ncnn"
    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")
    traced_model.save(f"{path}/model.pt")
    input_shape_str = json.dumps(input_shape)
    os.system(f"pnnx {path}/model.pt 'inputshape={input_shape_str}'")


    model_config["files"] = [
        f"model.ncnn.param",
        f"model.ncnn.bin",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

# comment/uncomment the ones you want to export.
# some exports may not work depending on the model or host operating system.
# openvino may require intel system.
# ncnn has limited model/op support.
export_openvino()
export_coreml()
export_onnx()
export_ncnn()
