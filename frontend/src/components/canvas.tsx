import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  WheelEventHandler,
} from "react";
import { MouseEvent as ReactMouseEvent } from "react";

export type Cursor =
  | "crosshair"
  | "default"
  | "not-allowed"
  | `custom-${string}`;

export type CanvasProps = {
  cursor?: Cursor;
  draw: (ctx: CanvasRenderingContext2D, frameCount: number) => void;
  onClick?: (x: number, y: number) => void;
  onMouseEnter?: (x: number, y: number) => void;
  onMouseLeave?: (x: number, y: number) => void;
  onMouseMove?: (x: number, y: number) => void;
  onMouseDown?: (x: number, y: number) => void;
  onMouseUp?: (x: number, y: number) => void;
  onWheel?: WheelEventHandler<HTMLCanvasElement>;
  className?: string;
  width?: number;
  height?: number;
};

const Canvas = ({
  cursor = "default",
  draw,
  className,
  width,
  height,
  onClick,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onMouseEnter,
  onMouseMove,
  onWheel,
}: CanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) return;
    const context = canvas.getContext("2d");

    if (!context) return;
    let frameCount = 0;
    let animationFrameId: number;

    const render = () => {
      frameCount++;
      draw(context, frameCount);
      animationFrameId = window.requestAnimationFrame(render);
    };

    render();

    return () => {
      window.cancelAnimationFrame(animationFrameId);
    };
  }, [canvasRef, draw]);

  const handleEvent = useCallback(
    (func?: (x: number, y: number) => void) =>
      (e: ReactMouseEvent<HTMLCanvasElement, MouseEvent>) => {
        if (!func) return;
        const canvas = canvasRef.current;

        if (!canvas) return;
        const context = canvas.getContext("2d");

        if (!context) return;

        const rect = canvas.getBoundingClientRect();

        // Calculate mouse coordinates relative to the canvas
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);

        func(x, y);
      },
    [canvasRef],
  );

  const cursorString = useMemo(() => {
    if (!cursor.startsWith("custom")) {
      return cursor;
    }

    return `url(${cursor.replace("custom-", "")}), auto`;
  }, [cursor]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      height={height}
      style={{
        cursor: cursorString,
      }}
      width={width}
      onClick={handleEvent(onClick)}
      onMouseDown={handleEvent(onMouseDown)}
      onMouseEnter={handleEvent(onMouseEnter)}
      onMouseLeave={handleEvent(onMouseLeave)}
      onMouseMove={handleEvent(onMouseMove)}
      onMouseUp={handleEvent(onMouseUp)}
      onWheel={onWheel}
    />
  );
};

export default Canvas;
