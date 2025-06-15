import createColorMap from "colormap";

export enum RoomType {
  LIVING_ROOM = 1,
  KITCHEN = 2,
  BEDROOM = 3,
  BATHROOM = 4,
  BALCONY = 5,
  ENTRANCE = 6,
  DINING_ROOM = 7,
  STUDY_ROOM = 8,
  STORAGE = 10,
  FRONT_DOOR = 15,
  UNKNOWN = 16,
  INTERIOR_DOOR = 17,
}

type colorMapFormat = "hex" | "rgbaString" | "rgba" | "float";
type colorMapType<T extends colorMapFormat> = T extends "rgba" | "float"
  ? Array<[number, number, number, number]>
  : string[];

type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;

export namespace RoomType {
  export function keys(): (keyof RoomType)[] {
    return Object.keys(RoomType).filter(
      (key) => !isNaN(Number(key)),
    ) as (keyof RoomType)[];
  }
  export function values(): number[] {
    return Object.values(RoomType).filter((value) => typeof value === "number");
  }

  export function entries(): [string, number][] {
    return Object.entries(RoomType).filter(
      ([key, value]) => isNaN(Number(key)) && typeof value === "number",
    ) as [string, number][];
  }

  export function toRestricted(type: RoomType): number {
    // enums = list(RoomType)
    // enums.remove(RoomType.INTERIOR_DOOR)
    // enums.remove(RoomType.FRONT_DOOR)
    // index_map = {room_type: i for i, room_type in enumerate(enums)}
    // return index_map[self]
    let enum_values = RoomType.values().filter(
      (value) =>
        value !== RoomType.FRONT_DOOR && value !== RoomType.INTERIOR_DOOR,
    );
    let index_map = Object.fromEntries(
      enum_values.map((value, index) => [value, index]),
    );

    return index_map[type];
  }

  export function restrictedLength(): number {
    return RoomType.length() - 2;
  }

  export function toMaskValue(type: RoomType | null): number {
    // if room_type is None:
    //     return -1
    // room_type_values = np.linspace(-1, 1, RoomType.restricted_length() + 1)
    // return room_type_values[room_type.index_restricted() + 1].item()
    if (type === null) {
      return -1;
    }
    const roomTypeValues = Array.from(
      { length: RoomType.restrictedLength() + 1 },
      (_, i) => -1 + (2 * i) / RoomType.restrictedLength(),
    );

    return roomTypeValues[RoomType.toRestricted(type) + 1];
  }

  export function keyToString(key: string): string {
    if (!key) {
      return "";
    }
    const words = key.split("_");

    return words
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  }

  export function length(): number {
    return RoomType.keys().length;
  }

  export class ColorMap<T extends colorMapFormat = "rgbaString"> {
    private readonly map: colorMapType<T>;
    constructor(
      private readonly format: T = "rgbaString" as T,
      private readonly alpha: number = 1,
      private readonly colormap: string = "hsv",
    ) {
      this.map = createColorMap({
        colormap: this.colormap,
        nshades: RoomType.length(),
        format: this.format,
        alpha: this.alpha,
      });
    }

    public getColor(type: number): ArrayElement<colorMapType<T>>;
    public getColor(type: number): any {
      return this.map[type];
    }
  }
}
