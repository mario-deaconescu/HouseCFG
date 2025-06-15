import { Select, SelectItem } from "@heroui/select";
import { useCallback, useMemo } from "react";
import { SharedSelection } from "@heroui/system";

import { Model, modelToName, useModel } from "@/context/model-context";

const modelOptions = Object.entries(Model).map(([key, value]) => ({
  key: key,
  value: value,
  label: modelToName[value],
}));

const ModelPicker = () => {
  const { model, setModel } = useModel();

  const selectModel = useCallback(
    (selection: SharedSelection) => {
      if (!(selection instanceof Set) || selection.size !== 1) {
        throw new Error("Invalid selection");
      }

      const value = selection.values().next().value;

      if (value === undefined) {
        throw new Error("Invalid selection");
      }

      setModel(value as Model);
    },
    [setModel],
  );

  const selectedKeys = useMemo(() => {
    return [model];
  }, [model]);

  return (
    <Select
      label={"Select Model"}
      selectedKeys={selectedKeys}
      onSelectionChange={selectModel}
    >
      {modelOptions.map((option) => (
        <SelectItem key={option.value}>{option.label}</SelectItem>
      ))}
    </Select>
  );
};

export default ModelPicker;
