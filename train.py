# train.py
import os
import json
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def build_model(num_classes, input_shape=(224,224,3), dropout=0.3):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    # freeze base
    for layer in base.layers:
        layer.trainable = False
    return model

def main(args):
    dataset_dir = args.dataset
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    epochs = args.epochs
    model_out = args.output  # e.g. currency_model.h5

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
    )

    num_classes = train_gen.num_classes
    print("Found classes:", train_gen.class_indices)

    # save mapping class_idx -> label
    class_map = {v: k for k, v in train_gen.class_indices.items()}
    with open('class_map.json', 'w') as f:
        json.dump(class_map, f)
    print("Saved class_map.json")

    model = build_model(num_classes, input_shape=(img_size[0], img_size[1], 3), dropout=args.dropout)

    # compile
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    ckpt = ModelCheckpoint('best_'+model_out, monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    # train (initially only top layers)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[ckpt, es, reduce_lr]
    )

    # fine-tune: unfreeze some layers and train lower LR
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=int(epochs/2),
        callbacks=[ckpt, es, reduce_lr]
    )

    # Save final model as HDF5 .h5
    model.save(model_out)  # e.g. currency_model.h5
    print("Saved model to", model_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset root folder')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output', type=str, default='currency_model.h5')
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
