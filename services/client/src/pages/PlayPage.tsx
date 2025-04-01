import React, { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import PlayerForm from "components/PlayerForm";
import GameScreen from "components/GameScreen";
import GameStateMessage from "components/GameStateMessage";
import BatteryStatus from "components/BatteryStatus";
import useGame, { GameState } from "hooks/useGame";
import { gameSocketURL, gameDurationSeconds } from "lib/config";

interface LocationState {
  autoStart?: boolean;
  playerName?: string;
}

const PlayPage = () => {
  const {
    playerName,
    gameMode,
    gameModes,
    gameState,
    formattedTime,
    health,
    score,
    startGame,
    error,
    battery,
  } = useGame(gameSocketURL, gameDurationSeconds);

  const location = useLocation();
  const locationState = (location.state as LocationState) || {};
  const [autoStarted, setAutoStarted] = useState(false);

  useEffect(() => {
    if (
      locationState.autoStart &&
      gameState === GameState.NotStarted &&
      !autoStarted
    ) {
      startGame(locationState.playerName || "Tutorial", gameMode);
      setAutoStarted(true);
    }
  }, [locationState, gameState, autoStarted, startGame, gameMode]);

  switch (gameState) {
    case GameState.Connecting:
    case GameState.Error:
    case GameState.Waiting:
    case GameState.NotStarted:
      return (
        <div className="flex flex-col gap-7 justify-center h-full">
          <PlayerForm
            gameModes={gameModes}
            isDisabled={gameState !== GameState.NotStarted}
            onSubmit={startGame}
          />
          <GameStateMessage state={gameState} errorMessage={error} />
          <BatteryStatus batteryPercentage={battery} />
        </div>
      );

      
      
      case GameState.InGame:
        return (
          <div className="flex flex-col items-center gap-6 p-4">
            <GameScreen
              playerName={playerName}
              gameMode={gameMode}
              health={health}
              score={score}
              time={formattedTime}
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-7xl">
              {/* MJPEG feed from Python container */}
              <div className="w-full">
                <h2 className="text-xl font-semibold mb-2 text-center text-orange-500">
                  Computer Camera Feed
                </h2>
                <img
                  src="http://localhost:5000/video_feed"
                  alt="Live Video Feed"
                  className="w-full rounded-xl border-2 border-gray-300 shadow-lg"
                />
              </div>
      
              {/* External feed via iframe */}
              <div className="w-full">
                <h2 className="text-xl font-semibold mb-2 text-center text-orange-500">
                  Rover Camera Feed
                </h2>
                <iframe
                  src="http://192.168.0.115:8081/index.html"
                  title="External Camera Feed"
                  className="w-full h-[480px] rounded-xl border-2 border-gray-300 shadow-lg"
                />
              </div>
            </div>
          </div>
        );
      
      
      
      
    case GameState.GameEnded:
      return (
        <Navigate
          to={`/leaderboard?player=${encodeURIComponent(
            playerName
          )}&gameMode=${gameMode}`}
        />
      );

    default:
      return null;
  }
};

export default PlayPage;
