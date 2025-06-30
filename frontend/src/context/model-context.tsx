import { createContext, useContext, useState } from "react";

enum Model {
  BUBBLES = "bubbles",
  BUBBLES_OLD = "bubbles_old",
  BUBBLES_V2 = "bubbles_v2",
  ROOM_TYPES = "room_types",
}

const modelToName: Record<Model, string> = {
  [Model.BUBBLES]: "Bubbles",
  [Model.ROOM_TYPES]: "Room Types",
  [Model.BUBBLES_OLD]: "Bubbles Old",
  [Model.BUBBLES_V2]: "Bubbles V2",
};

type ModelContextType = {
  model: Model;
  setModel: (model: Model) => void;
};

const ModelContext = createContext<ModelContextType>({
  model: Model.BUBBLES,
  setModel: () => {},
});

const ModelProvider = ({ children }: { children: React.ReactNode }) => {
  const [model, setModel] = useState<Model>(Model.BUBBLES);

  return (
    <ModelContext.Provider value={{ model, setModel }}>
      {children}
    </ModelContext.Provider>
  );
};

const useModel = () => useContext(ModelContext);

export default ModelProvider;

export { Model, ModelContext, useModel, modelToName };
