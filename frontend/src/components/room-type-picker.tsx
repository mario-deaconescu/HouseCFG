import { useCallback, useState } from "react";
import { NumberInput } from "@heroui/number-input";

import { RoomType } from "@/types/room-type.ts";
import { defaultMap, validRoomTypes } from "@/types/room-type-picker.ts";

type Props = {
  onChange: (roomTypes: Map<RoomType, number>) => void;
};

const RoomTypePicker = ({ onChange }: Props) => {
  const [roomTypes, setRoomTypes] = useState<Map<RoomType, number>>(defaultMap);

  const handleChange = useCallback(
    (key: RoomType, value: number) => {
      setRoomTypes((prev) => {
        const newMap = new Map(prev);

        newMap.set(key, value);

        return newMap;
      });
      onChange(roomTypes);
    },
    [onChange, roomTypes],
  );

  return (
    <div className={"flex flex-col w-full gap-1 aspect-square justify-between"}>
      {validRoomTypes.map(([key, value]) => (
        <div key={value} className={"flex flex-row items-center gap-5"}>
          <NumberInput
            aria-label={RoomType.keyToString(key)}
            className={"w-[10rem]"}
            minValue={0}
            size={"sm"}
            value={roomTypes.get(value)}
            variant={"bordered"}
            onChange={(newValue) =>
              handleChange(value, typeof newValue === "number" ? newValue : 0)
            }
          />
          <p>{RoomType.keyToString(key)}</p>
        </div>
      ))}
    </div>
  );
};

export default RoomTypePicker;
