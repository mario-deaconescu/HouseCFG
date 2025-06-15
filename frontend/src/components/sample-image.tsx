import { useEffect, useRef } from "react";

import { SampleImage } from "@/types/sample.ts";

type Props = {
  sample: SampleImage;
  scaleFactor: number;
  className?: string;
  canvasSize: number;
  removeBackground?: boolean;
};

const SampleImageCanvas = ({
  sample,
  scaleFactor,
  className,
  canvasSize,
  removeBackground = false,
}: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    if (!ctx) return;

    const imageData = ctx.createImageData(
      canvasSize * scaleFactor,
      canvasSize * scaleFactor,
    );
    const buffer = imageData.data; // Uint8ClampedArray

    for (let i = 0; i < canvasSize; i++) {
      for (let j = 0; j < canvasSize; j++) {
        for (let x = 0; x < scaleFactor; x++) {
          for (let y = 0; y < scaleFactor; y++) {
            const index =
              ((i * scaleFactor + x) * canvasSize * scaleFactor +
                (j * scaleFactor + y)) *
              4;

            buffer[index] = sample.image[i][j][0];
            buffer[index + 1] = sample.image[i][j][1];
            buffer[index + 2] = sample.image[i][j][2];
            const total = sample.image[i][j].reduce((a, b) => a + b, 0);

            buffer[index + 3] = total <= 0 && removeBackground ? 0 : 255; // alpha channel
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    // ctx.globalCompositeOperation = "copy";
    // ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height);
  }, [canvasRef, sample, scaleFactor, canvasSize, removeBackground]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      height={canvasSize * scaleFactor}
      width={canvasSize * scaleFactor}
    />
  );
};

export default SampleImageCanvas;
