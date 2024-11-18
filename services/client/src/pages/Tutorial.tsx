import React from "react";
import { Link } from "react-router-dom";
import { ReactComponent as Combomark } from "assets/openliberty_combomark.svg";

const TutorialPage = () => {
  return (
    <div className="flex flex-col items-center justify-center mx-auto h-full">
      <div className="flex flex-col items-center">
        <Combomark className="h-24 mr-16" />
        <p className="text-orange text-3xl">Tutorial Mode</p>
      </div>
      <div className="mt-10 w-3/4 bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold mb-4">Getting Started</h2>
        <p className="mb-6">
          Welcome to the Space Rover Mission tutorial! This guide will help you understand the gameplay, objectives, and controls so you can get the best out of your mission.
        </p>
        
        <h3 className="text-xl font-semibold mb-3">Objective</h3>
        <p className="mb-6">
          The rover will automatically navigate through various terrains, avoid obstacles, and complete tasks to score points
        </p>
      </div>
      <Link
        className="block text-5xl font-medium px-32 py-8 my-14 rounded-lg bg-green hover:bg-green-light"
        to="/play"
        state={{ playerName: "Tutorial", autoStart: true, initialGameState: "InGame" }}
      >
        Start Mission
      </Link>
    </div>
  );
};

export default TutorialPage;