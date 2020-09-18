from shutil import copyfile
import random
import pathlib



def get_keras_datasets(origin_dir):

  gener_train = ImageDataGenerator(rescale=1/255,
                    # rotation_range=40,
                    # width_shift_range=0.2,
                    # height_shift_range=0.2,
                    # shear_range=0.2,
                    # zoom_range=0.2,
                    # horizontal_flip=True,
                    # fill_mode='nearest'
                    )

  gener_val = ImageDataGenerator(rescale=1/255)

  train_gen = gener_train.flow_from_directory(
      origin_dir+'train',
      batch_size=128,
      target_size=(192,192),
      class_mode='categorical'
  )

  val_gen = gener_train.flow_from_directory(
      origin_dir+'val',
      batch_size=128,
      target_size=(192,192),
      class_mode='categorical'
  )
  return train_gen,val_gen


def get_tf_dataset(origin_dir,preprocess_image,batch_size=32,target_size=[224,224]):
  data_root = pathlib.Path(origin_dir)
  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]
  print(len(all_image_paths))
  label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
  label_to_index = dict((name,index) for index,name in enumerate(label_names))
  all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
              for path in all_image_paths]
  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)

  image_ds = path_ds.map(preprocess_image)
  image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))

  ds = image_label_ds.shuffle(
      buffer_size=len(all_image_paths)).repeat().batch(batch_size)
  steps_per_epoch = tf.math.ceil(len(all_image_paths)/batch_size).numpy()
  return ds,steps_per_epoch


def get_tf_datasets(origin_dir,train_size=0.75,batch_size=32,target_size=[224,224]):
  # use code below to import 
  # from shutil import copyfile
  # import random
  # import pathlib
  def preprocess_image_train(path,target_size=[224,224]):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,target_size)
    image /= 255.0
    return image
  def preprocess_image_val(path,target_size=[224,224]):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,target_size)
    image /= 255.0
    return image
  #mkdirs
  train = './dataset/train/'
  val = './dataset/val/'
  if not os.path.exists('./dataset'):
    os.mkdir('./dataset/')
  if not os.path.exists(train):
    os.mkdir(train)
  if not os.path.exists(val):
    os.mkdir(val)

  list1 = os.listdir(origin_dir)
  for each in list1:
    if not os.path.exists(train+each):
      os.mkdir(train+each)
    if not os.path.exists(val+each):
      os.mkdir(val+each)
  #get all dirs
  data_root = pathlib.Path(origin_dir)
  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]

  paths_to_shuffle = random.sample(all_image_paths,len(all_image_paths))
  split_pos = int(train_size*len(all_image_paths))
  train_paths = paths_to_shuffle[:split_pos]
  val_paths = paths_to_shuffle[split_pos:]

  for each in train_paths:
    parts = each.split('/')
    target = parts[-2]+'/'+parts[-1]
    copyfile(each,train+target)
  for each in val_paths:
    parts = each.split('/')
    target = parts[-2]+'/'+parts[-1]
    copyfile(each,val+target)
  # return但是我不想return

  #train dataset
  train_ds,steps_per_epoch = get_tf_dataset('./dataset/train',preprocess_image_train,batch_size=batch_size,target_size=target_size)
  #val dataset
  val_ds,validation_steps = get_tf_dataset('./dataset/val',preprocess_image_val,batch_size=batch_size,target_size=target_size)

  return train_ds , val_ds , steps_per_epoch , validation_steps
