// Built by: (Benjamin Lefebvre - 301234587)

import { useEffect, useState } from "react";
import "./styles/App.css";
import axios from "axios";

function App() {
  const [options, setOptions] = useState({
    DIVISION: [],
    LOCATION_TYPE: [],
    PREMISES_TYPE: [],
    NEIGHBOURHOOD_158: [],
    BIKE_TYPE: [],
  });

  const [data, setData] = useState({
    OCC_YEAR: "",
    OCC_MONTH: "",
    OCC_DOW: "",
    DIVISION: "",
    LOCATION_TYPE: "",
    PREMISES_TYPE: "",
    BIKE_TYPE: "",
    BIKE_COST: "",
    NEIGHBOURHOOD_158: "",
  });

  const [prediction, setPrediction] = useState("");

  // Years from today to 20 years prior
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 21 }, (_, i) => currentYear - i);

  useEffect(() => {
    // Fetch options from API
    axios
      .get("http://127.0.0.1:5000/options", {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": true,
        },
      })
      .then((response) => {
        setOptions(response.data);
        console.log(response.data);
      })
      .catch((error) => {
        console.error("There was an error fetching the options:", error);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    console.log(data);
  }, [data]);

  function handleChange(
    e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>
  ) {
    const { name, value } = e.target;
    setData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  }

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    console.log(data);

    // Validate the data
    if (Object.values(data).some((value) => value === "")) {
      alert("Please fill in all the fields");
      return;
    }

    // Make a POST request to the API
    axios.post("http://127.0.0.1:5000/predict", data).then((response) => {
      console.log(response.data);
      setPrediction(response.data.prediction);
    });

    // Clear the form
  }

  return (
    <>
      <h1>Police Toronto - Bike Prediction App</h1>

      <div className='dashboard'>
        <div className='input-form tile'>
          <h2>Details about the stolen bike</h2>
          <form onSubmit={handleSubmit}>
            {/* YEAR - OCC_YEAR */}
            <label htmlFor='OCC_YEAR'>What year was the bike was stolen?</label>
            <select
              name='OCC_YEAR'
              id='OCC_YEAR'
              value={data.OCC_YEAR}
              onChange={handleChange}
            >
              <option value=''>Select Year</option>
              {years.map((year) => (
                <option key={year} value={year}>
                  {year}
                </option>
              ))}
            </select>

            {/* MONTH - OCC_MONTH */}
            <label htmlFor='OCC_MONTH'>
              What month was the bike was stolen?
            </label>
            <select
              name='OCC_MONTH'
              id='OCC_MONTH'
              value={data.OCC_MONTH}
              onChange={handleChange}
            >
              <option value=''>Select Month</option>
              <option value='January'>January</option>
              <option value='February'>February</option>
              <option value='March'>March</option>
              <option value='April'>April</option>
              <option value='May'>May</option>
              <option value='June'>June</option>
              <option value='July'>July</option>
              <option value='August'>August</option>
              <option value='September'>September</option>
              <option value='October'>October</option>
              <option value='November'>November</option>
              <option value='December'>December</option>
            </select>

            {/* DAY OF THE WEEK - OCC_DOW */}
            <label htmlFor='OCC_DOW'>
              What day of the week was the bike was stolen?
            </label>
            <select
              name='OCC_DOW'
              id='OCC_DOW'
              value={data.OCC_DOW}
              onChange={handleChange}
            >
              <option value=''>Select Day</option>
              <option value='Sunday'>Sunday</option>
              <option value='Monday'>Monday</option>
              <option value='Tuesday'>Tuesday</option>
              <option value='Wednesday'>Wednesday</option>
              <option value='Thursday'>Thursday</option>
              <option value='Friday'>Friday</option>
              <option value='Saturday'>Saturday</option>
            </select>

            {/* Police Division - DIVISION */}
            <label htmlFor='DIVISION'>
              Which police division is in charge of the case?
            </label>
            <select
              name='DIVISION'
              id='DIVISION'
              value={data.DIVISION}
              onChange={handleChange}
            >
              <option value=''>Select Division</option>
              {options.DIVISION.map((division) => (
                <option key={division} value={division}>
                  {division}
                </option>
              ))}
            </select>

            {/* Location Type - LOCATION_TYPE */}
            <label htmlFor='LOCATION_TYPE'>
              What location type describe the best the last place your bike was
              seen?
            </label>
            <select
              name='LOCATION_TYPE'
              id='LOCATION_TYPE'
              value={data.LOCATION_TYPE}
              onChange={handleChange}
            >
              <option value=''>Select Location Type</option>
              {options.LOCATION_TYPE.map((locationType) => (
                <option key={locationType} value={locationType}>
                  {locationType}
                </option>
              ))}
            </select>

            {/* Premises type - PREMISES_TYPE */}
            <label htmlFor='PREMISES_TYPE'>What premises type was it?</label>
            <select
              name='PREMISES_TYPE'
              id='PREMISES_TYPE'
              value={data.PREMISES_TYPE}
              onChange={handleChange}
            >
              <option value=''>Select Premises Type</option>
              {options.PREMISES_TYPE.map((premisesType) => (
                <option key={premisesType} value={premisesType}>
                  {premisesType}
                </option>
              ))}
            </select>

            {/* Bike Cost - BIKE_COST */}
            <label htmlFor='BIKE_COST'>How much is the bike cost?</label>
            <input
              type='number'
              name='BIKE_COST'
              id='BIKE_COST'
              min={0}
              max={100000}
              value={data.BIKE_COST}
              onChange={handleChange}
            />

            {/* Neiboorhood - NEIGHBOURHOOD_158 */}
            <label htmlFor='NEIGHBOURHOOD_158'>
              What neiboorhood was it stolen from?
            </label>
            <select
              name='NEIGHBOURHOOD_158'
              id='NEIGHBOURHOOD_158'
              value={data.NEIGHBOURHOOD_158}
              onChange={handleChange}
            >
              <option value=''>Select Neighbourhood</option>
              {options.NEIGHBOURHOOD_158.map((neighbourhood) => (
                <option key={neighbourhood} value={neighbourhood}>
                  {neighbourhood}
                </option>
              ))}
            </select>

            {/* Bike Type - BIKE_TYPE */}
            <label htmlFor='BIKE_TYPE'>What type of bike was it?</label>
            <select
              name='BIKE_TYPE'
              id='BIKE_TYPE'
              value={data.BIKE_TYPE}
              onChange={handleChange}
            >
              <option value=''>Select Bike Type</option>
              {options.BIKE_TYPE.map((bikeType) => (
                <option key={bikeType} value={bikeType}>
                  {bikeType}
                </option>
              ))}
            </select>

            <button type='submit'>Predict</button>
          </form>
        </div>

        <div className='prediction-results tile'>
          <h2>Outcome Predictions</h2>
          <p>Is your bike likely to be returned or not?</p>
          {/* Your bike is likely to be (stolen or returned) */}
          {prediction && <p>Your bike is likely to be {prediction}</p>}
        </div>
      </div>
    </>
  );
}

export default App;
