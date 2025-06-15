import { RoomType } from "@/types/room-type.ts";
import { canvasSize } from "@/types/canvas.ts";
import { Mask } from "@/types/mask.ts";

export class Bubble {
  constructor(
    public origin: [number, number],
    public radius: number,
    public type: RoomType,
  ) {}

  public intersectsPoint(x: number, y: number): boolean {
    const dx = x - this.origin[0];
    const dy = y - this.origin[1];

    return dx * dx + dy * dy < this.radius * this.radius;
  }

  public touchesPoint(x: number, y: number): boolean {
    const dx = x - this.origin[0];
    const dy = y - this.origin[1];

    return dx * dx + dy * dy === this.radius * this.radius;
  }

  public intersectsBubble(other: Bubble): boolean {
    const dx = this.origin[0] - other.origin[0];
    const dy = this.origin[1] - other.origin[1];

    return (
      dx * dx + dy * dy <
      (this.radius + other.radius) * (this.radius + other.radius)
    );
  }

  public isOutOfBounds(): boolean {
    return (
      this.origin[0] - this.radius < 0 ||
      this.origin[0] + this.radius >= canvasSize ||
      this.origin[1] - this.radius < 0 ||
      this.origin[1] + this.radius >= canvasSize
    );
  }
}

export class BubbleMask {
  constructor(public bubbles: Bubble[]) {}

  public toMask(): Mask<number> {
    let mask = Array(canvasSize);

    for (let i = 0; i < canvasSize; i++) {
      mask[i] = Array(canvasSize).fill(-1);
    }

    for (const bubble of this.bubbles) {
      const [x, y] = bubble.origin;
      const radius = bubble.radius;

      for (let i = Math.ceil(x - radius); i <= x + radius; i++) {
        for (let j = Math.ceil(y - radius); j <= y + radius; j++) {
          if (bubble.intersectsPoint(i, j) || bubble.touchesPoint(i, j)) {
            // console.log(i, j);
            mask[i][j] = RoomType.toMaskValue(bubble.type);
          }
        }
      }
    }

    return mask;
  }
}

export const enum BubblesMode {
  CREATE,
  DELETE,
}
