import React from "react";
import { Link } from "react-router-dom";
import { LeaderboardEntry } from "hooks/useLeaderboard";
import { formatTime } from "lib/utils";

type Props = LeaderboardEntry;

const PlacementDisplay = ({ rank, player, time, score }: Props) => {
  return (
    <div className="text-gray-50 text-center pt-5 pb-10">
      <h1 className="text-green text-5xl">Mission completed</h1>
      <div className="flex gap-20 justify-center p-5">
        <div>
          <p className="text-gray-400">Rank</p>
          <p className="text-7xl text-semibold">{rank}</p>
        </div>
        <div>
          <p className="text-gray-400">Time</p>
          <p className="text-7xl text-semibold">{formatTime(time)}</p>
        </div>
        <div>
          <p className="text-gray-400">Score</p>
          <p className="text-7xl text-semibold">{score}</p>
        </div>
      </div>
      <p className="text-3xl">
        Thank you for playing, <span className="text-orange">{player}</span>
      </p>
      <Link
        className="inline-block mt-5 px-10 py-5 bg-green hover:bg-green-light text-black rounded-full text-2xl"
        to="/play"
      >
        Play again
      </Link>
    </div>
  );
};

export default PlacementDisplay;
