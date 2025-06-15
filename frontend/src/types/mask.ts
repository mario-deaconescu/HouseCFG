import { Dispatch, SetStateAction } from "react";

import { canvasSize } from "@/types/canvas.ts";

export type Mask<T> = T[][];

export const transpose = <T>(mask: Mask<T>): Mask<T> => {
  return mask[0].map((_, i) => mask.map((row) => row[i]));
};

export const enum MaskMode {
  BRUSH,
  ERASER,
  FILL,
  FILL_ERASER,
}

export const defaultMask = <T>(
  width: number,
  height: number,
  value: T,
): Mask<T> => {
  return Array(width).fill(Array(height).fill(value));
};

export const updateArray = <T>(
  setter: Dispatch<SetStateAction<Mask<T>>>,
  i: number,
  j: number,
  value: T,
) => {
  setter((prevMask: Mask<T>): Mask<T> => {
    const newMask = [...prevMask];

    newMask[i] = [...newMask[i]];
    newMask[i][j] = value;

    return newMask;
  });
};

export const fillArray = <T>(
  setter: Dispatch<SetStateAction<Mask<T>>>,
  start: [number, number],
  end: [number, number],
  value: T,
) => {
  setter((prevMask: Mask<T>): Mask<T> => {
    const newMask = [...prevMask];

    for (let i = start[0]; i <= end[0]; i++) {
      newMask[i] = [...newMask[i]];
      for (let j = start[1]; j <= end[1]; j++) {
        newMask[i][j] = value;
      }
    }

    return newMask;
  });
};

export const forEachInMask = (func: (i: number, j: number) => void) => {
  for (let i = 0; i < canvasSize; i++) {
    for (let j = 0; j < canvasSize; j++) {
      func(i, j);
    }
  }
};
