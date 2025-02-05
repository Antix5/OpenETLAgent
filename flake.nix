{
  description = "Open ETL Agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # --- Python Environment Setup ---
        pythonEnv = import ./python-env.nix {
          inherit pkgs;
          python = pkgs.python311;

        };

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv.out.venv
          ];

          shellHook = ''
            source .env

            # --- Python Environment Activation ---
            source ${pythonEnv.out.activationScript}/bin/activate-venv
            echo "Virtual environment activated. To deactivate, run 'deactivate'."

          '';
        };
      });
}