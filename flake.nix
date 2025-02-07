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
        python = pkgs.python311;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (import ./python-env.nix { inherit pkgs python; }).venv
          ];

          shellHook = ''
            source .env
            echo "Python environment ready. Activate with:"
            echo "source ${(import ./python-env.nix { inherit pkgs python; }).venv}/bin/activate"
          '';
        };
      });
}