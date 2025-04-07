"use client";
import Select from "react-select";

export function MultiSelect({ options, selected, onChange }) {
  return (
    <Select
      isMulti
      options={options}
      value={selected}
      onChange={onChange}
      className="w-full"
      styles={{
        control: (base) => ({
          ...base,
          backgroundColor: "white",
          borderColor: "#ccc",
        }),
      }}
    />
  );
}
