{ pkgs, python }:
let
  pythonPackages = python.pkgs;
in
pkgs.stdenv.mkDerivation rec {
  name = "python-env";
  src = ./.;

  venv = pkgs.stdenv.mkDerivation {
    name = "open-etl-venv";
    buildInputs = [ python ];

    phases = [ "installPhase" ];

    installPhase = ''
      # Create venv in writable temporary directory
      WORKDIR=$(mktemp -d)
      ${python}/bin/python -m venv $WORKDIR/venv
      source $WORKDIR/venv/bin/activate
      pip install --no-cache-dir \
        tqdm>=4.0 \
        pydantic>=2.5,<3 \
        numpy>=1.26,<2 \
        pandas>=2.1,<3 \
        python-dotenv>=1,<2 \
        requests>=2.31,<3 \
        pytest>=8.3.4 \
        setuptools==75.8.0 \
        pydantic-ai>=0.0.22

      # Copy the finished venv to Nix store
      mkdir -p $out
      cp -r $WORKDIR/venv/* $out/
    '';
  };
}
