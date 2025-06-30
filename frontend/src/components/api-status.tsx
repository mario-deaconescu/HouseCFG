import { useEffect, useState } from "react";
import { Spinner } from "@heroui/spinner";
import { Checkbox } from "@heroui/checkbox";

import apiBase from "@/config/api.ts";

interface Props {
  apiUrl?: string;
}

const ApiStatus = ({ apiUrl = `${apiBase}/health` }: Props) => {
  const [status, setStatus] = useState<boolean | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(apiUrl);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      setStatus(data.status);
    };

    eventSource.onerror = () => {
      setStatus(false);
    };

    return () => eventSource.close();
  }, [apiUrl]);

  return status === null ? (
    <div className={"flex flex-row items-center justify-center gap-2"}>
      <Spinner variant={"dots"} />
      <span className={"mt-1"}>Connecting</span>
    </div>
  ) : (
    <Checkbox
      className={`pointer-events-none flex flex-row items-center justify-center ${status ? "text-green-500" : "text-red-500"}`}
      // isDisabled={true}
      isSelected={status}
    >
      {status ? "Connected" : "Disconnected"}
    </Checkbox>
  );
};

export default ApiStatus;
