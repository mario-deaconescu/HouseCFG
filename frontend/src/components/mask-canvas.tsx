import { useCallback, useEffect, useState } from "react";

import Canvas from "./canvas";

import {
  fillArray,
  forEachInMask,
  Mask,
  MaskMode,
  updateArray,
} from "@/types/mask";
import { canvasSize } from "@/types/canvas.ts";

type Props = {
  className?: string;
  scaleFactor?: number;
  mode: MaskMode;
  onChange: (mask: Mask<boolean>) => void;
};

const MaskCanvas = ({ className, scaleFactor = 1, mode, onChange }: Props) => {
  const [mask, setMask] = useState<Mask<boolean>>(
    Array(canvasSize).fill(Array(canvasSize).fill(false)),
  );
  const [hovered, setHovered] = useState<Mask<boolean>>(
    Array(canvasSize).fill(Array(canvasSize).fill(false)),
  );
  const [fillHovered, setFillHovered] = useState<Mask<boolean>>(
    Array(canvasSize).fill(Array(canvasSize).fill(false)),
  );
  const [holding, setHolding] = useState(false);
  const [startFill, setStartFill] = useState<[number, number] | null>(null);

  const resetFill = useCallback(() => {
    setStartFill(null);
    setFillHovered(Array(canvasSize).fill(Array(canvasSize).fill(false)));
  }, [setStartFill, setFillHovered]);

  useEffect(() => {
    if (mode !== MaskMode.FILL && mode !== MaskMode.FILL_ERASER) {
      resetFill();
    }
  }, [mode]);

  const updateMask = useCallback(
    (i: number, j: number, filled: boolean) => {
      if (i < 0 || i >= canvasSize || j < 0 || j >= canvasSize) {
        return;
      }
      updateArray(setMask, i, j, filled);
    },
    [setMask],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      forEachInMask((i, j) => {
        if (mask[i][j]) {
          ctx.fillStyle = "black";
          ctx.fillRect(
            i * scaleFactor,
            j * scaleFactor,
            scaleFactor,
            scaleFactor,
          );
        } else {
          ctx.clearRect(
            i * scaleFactor,
            j * scaleFactor,
            scaleFactor,
            scaleFactor,
          );
        }
      });

      forEachInMask((i, j) => {
        if (fillHovered[i][j]) {
          switch (mode) {
            case MaskMode.FILL:
              ctx.fillStyle = "rgba(0, 255, 0, 0.5)";
              break;
            case MaskMode.FILL_ERASER:
              ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
              break;
            default:
              ctx.fillStyle = "black";
          }
          ctx.fillRect(
            i * scaleFactor,
            j * scaleFactor,
            scaleFactor,
            scaleFactor,
          );
        }
      });

      forEachInMask((i: number, j) => {
        if (hovered[i][j]) {
          switch (mode) {
            case MaskMode.BRUSH:
              ctx.strokeStyle = "blue";
              break;
            case MaskMode.ERASER:
              ctx.strokeStyle = "red";
              break;
            case MaskMode.FILL:
              ctx.strokeStyle = "green";
              break;
            case MaskMode.FILL_ERASER:
              ctx.strokeStyle = "orange";
              break;
            default:
              ctx.strokeStyle = "black";
          }
          ctx.lineWidth = 2;
          ctx.strokeRect(
            i * scaleFactor,
            j * scaleFactor,
            scaleFactor,
            scaleFactor,
          );
        }
      });

      // Draw grid
      ctx.strokeStyle = "rgba(0, 0, 0, 0.2)";
      ctx.lineWidth = 1;
      for (let i = 0; i < canvasSize; i++) {
        ctx.beginPath();
        ctx.moveTo(i * scaleFactor, 0);
        ctx.lineTo(i * scaleFactor, canvasSize * scaleFactor);
        ctx.stroke();
      }
      for (let j = 0; j < canvasSize; j++) {
        ctx.beginPath();
        ctx.moveTo(0, j * scaleFactor);
        ctx.lineTo(canvasSize * scaleFactor, j * scaleFactor);
        ctx.stroke();
      }
    },
    [scaleFactor, mask, hovered, mode, fillHovered],
  );

  const toCanvasRef = useCallback(
    (x: number, y: number) => {
      return [Math.floor(x / scaleFactor), Math.floor(y / scaleFactor)];
    },
    [scaleFactor],
  );

  const onClick = useCallback(
    (x: number, y: number) => {
      if (mode !== MaskMode.FILL && mode !== MaskMode.FILL_ERASER) {
        return;
      }
      const [i, j] = toCanvasRef(x, y);

      if (startFill === null) {
        setStartFill([i, j]);
      } else {
        let [startI, startJ] = startFill;
        let [endI, endJ] = [i, j];

        if (endI < startI) {
          [startI, endI] = [endI, startI];
        }
        if (endJ < startJ) {
          [startJ, endJ] = [endJ, startJ];
        }
        fillArray(
          setMask,
          [startI, startJ],
          [endI, endJ],
          mode === MaskMode.FILL,
        );
        resetFill();
      }
    },
    [
      scaleFactor,
      mode,
      startFill,
      setStartFill,
      setMask,
      mask,
      resetFill,
      onChange,
    ],
  );

  const onMouseMove = useCallback(
    (x: number, y: number) => {
      const [i, j] = toCanvasRef(x, y);

      if (i < 0 || i >= canvasSize || j < 0 || j >= canvasSize) {
        return;
      }

      const hovered = Array(canvasSize).fill(Array(canvasSize).fill(false));

      hovered[i] = [...hovered[i]];
      hovered[i][j] = true;
      setHovered(hovered);

      if (holding && (mode === MaskMode.BRUSH || mode === MaskMode.ERASER)) {
        const filled = mode === MaskMode.BRUSH;
        const [i, j] = toCanvasRef(x, y);

        updateMask(i, j, filled);
      }

      if (startFill !== null) {
        const fillHovered = Array(canvasSize).fill(
          Array(canvasSize).fill(false),
        );
        let [startI, startJ] = startFill;
        let [endI, endJ] = [i, j];

        if (endI < startI) {
          [startI, endI] = [endI, startI];
        }
        if (endJ < startJ) {
          [startJ, endJ] = [endJ, startJ];
        }
        for (let x = startI; x <= endI; x++) {
          for (let y = startJ; y <= endJ; y++) {
            fillHovered[x] = [...fillHovered[x]];
            fillHovered[x][y] = true;
          }
        }
        setFillHovered(fillHovered);
      }
    },
    [scaleFactor, setHovered, holding, startFill],
  );

  const onMouseLeave = useCallback(() => {
    setHovered(Array(canvasSize).fill(Array(canvasSize).fill(false)));
    resetFill();
    setHolding(false);
  }, [scaleFactor, setHovered]);

  const onMouseDown = useCallback(() => setHolding(true), [setHolding]);
  const onMouseUp = useCallback(() => setHolding(false), [setHolding]);

  useEffect(() => onChange && onChange(mask), [mask, onChange]);

  return (
    <Canvas
      className={className}
      cursor={"crosshair"}
      draw={draw}
      height={canvasSize * scaleFactor}
      width={canvasSize * scaleFactor}
      onClick={onClick}
      onMouseDown={onMouseDown}
      onMouseLeave={onMouseLeave}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    />
  );
};

export default MaskCanvas;
