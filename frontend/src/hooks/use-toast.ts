import { addToast } from "@heroui/toast";

const useToast = () => {
  const makeCallback =
    (
      color:
        | "default"
        | "primary"
        | "foreground"
        | "secondary"
        | "success"
        | "warning"
        | "danger",
    ) =>
    (title: string, description?: string) =>
      addToast({
        title,
        description,
        color,
      });

  return {
    defaultToast: makeCallback("default"),
    primaryToast: makeCallback("primary"),
    foregroundToast: makeCallback("foreground"),
    secondaryToast: makeCallback("secondary"),
    successToast: makeCallback("success"),
    warningToast: makeCallback("warning"),
    dangerToast: makeCallback("danger"),
    errorToast: makeCallback("danger").bind(null, "Error"),
  };
};

export default useToast;
